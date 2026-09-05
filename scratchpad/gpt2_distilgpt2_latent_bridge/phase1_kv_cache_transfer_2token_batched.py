#!/usr/bin/env python3
"""KV-cache handoff, 2-token (6-digit) fact -- bandwidth step 2 on the
mechanism that actually works.

The embedding-prefix mechanism collapsed sharply from 81.25% (1 token) to
0% (2 tokens) at 64 training examples, later shown to be a sample-
complexity artifact (512 examples recovered 59.38%), not a hard channel
ceiling. The KV-cache handoff mechanism just reached 100% top-1 / 27.787
nats gap on the 1-token task with a full-batch-trained low-rank adapter
(phase1_kv_cache_transfer_trained_batched.py) -- a categorically stronger
starting point. This tests whether that mechanism also degrades sharply
at 2 tokens, or whether the earlier collapse really was specific to the
weaker embedding-prefix channel/optimization regime.

Same architecture as the 1-token batched KV script: GPT-2's KV-cache for
"The secret code is {6-digit code}." transplanted into DistilGPT2 via
past_key_values at layers (1,3,5,7,9,11), low-rank residual adapter
(A:(768,8), B:(768,8)->zero-init, no bias) per layer per K/V stream,
full-batch training (one gradient step per epoch over all 64 train
documents, not per-document SGD -- established as the categorically
better optimization regime for this class of adapter). Codes are 6-digit
numbers verified to tokenize as exactly 2 BPE tokens (~21% of random
6-digit numbers qualify, checked directly). Scoring uses the standard
scored-window teacher-forcing convention (receiver fed suffix + both real
target tokens, logits at the last 2 positions predict them) -- same
convention as phase1_controlled_fact_transfer_2token.py.

Primary gate unchanged: correct-vs-shuffled paired bootstrap CI > 0.
Secondary: top1_exact (both tokens correct).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

SEED = 20260904
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_CODES = 64
DEV_CODES = 32
SCORED = 2
CODE_MIN, CODE_MAX = 100000, 999999
CANDIDATE_SCAN_SEED = 424242
CANDIDATE_POOL_TARGET = 600
SENDER_LAYER_SELECTION = (1, 3, 5, 7, 9, 11)
RANK = 8
A_INIT_STD = 0.02
MAX_EPOCHS = 300
DEV_EVAL_EVERY = 1
STILL_IMPROVING_REL_THRESHOLD = 0.01
LR = 1e-2


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bucket(code: int) -> str:
    value = int(sha(str(code).encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def two_token_codes(tokenizer, pool_target: int) -> list[int]:
    rng = random.Random(CANDIDATE_SCAN_SEED)
    seen: set[int] = set()
    eligible: list[int] = []
    while len(eligible) < pool_target:
        c = rng.randint(CODE_MIN, CODE_MAX)
        if c in seen:
            continue
        seen.add(c)
        if len(tokenizer.encode(f" {c}", add_special_tokens=False)) == 2:
            eligible.append(c)
    return eligible


def select_codes(eligible: list[int], split: str, count: int) -> list[int]:
    codes = sorted((c for c in eligible if bucket(c) == split), key=lambda c: sha(str(c).encode()))
    if len(codes) < count:
        raise RuntimeError(f"only {len(codes)} eligible {split} codes")
    return codes[:count]


def build_batch(sender, tokenizer, codes: list[int]):
    """Single batched sender forward pass, not a per-document loop -- fixed
    after the 2-key script's per-document data prep turned out to dominate
    wall time despite the training loop itself being batched."""
    texts = [f"The secret code is {code}." for code in codes]
    tokenized = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    lens = {len(t) for t in tokenized}
    assert len(lens) == 1, f"prefix length not constant: {lens}"
    ids = torch.tensor(tokenized, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = sender(ids, use_cache=True)
    pkv = out.past_key_values
    keys = [pkv.layers[i].keys.clone().detach() for i in SENDER_LAYER_SELECTION]
    values = [pkv.layers[i].values.clone().detach() for i in SENDER_LAYER_SELECTION]

    target_rows = []
    for code in codes:
        target_ids = tokenizer.encode(f" {code}", add_special_tokens=False)
        assert len(target_ids) == SCORED
        target_rows.append(target_ids)
    target = torch.tensor(target_rows, dtype=torch.long, device=DEVICE)  # (batch, SCORED)
    return keys, values, target


class LowRankKVAdapter(nn.Module):
    def __init__(self, n_layers: int, head_dim: int, n_head: int, rank: int, seed: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        width = n_head * head_dim
        g = torch.Generator().manual_seed(seed)
        self.A_k = nn.ParameterList([nn.Parameter(torch.randn(width, rank, generator=g) * A_INIT_STD) for _ in range(n_layers)])
        self.B_k = nn.ParameterList([nn.Parameter(torch.zeros(rank, width)) for _ in range(n_layers)])
        self.A_v = nn.ParameterList([nn.Parameter(torch.randn(width, rank, generator=g) * A_INIT_STD) for _ in range(n_layers)])
        self.B_v = nn.ParameterList([nn.Parameter(torch.zeros(rank, width)) for _ in range(n_layers)])

    def _lowrank_correct(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(b * s, h * d)
        correction = (flat @ A) @ B
        out_flat = flat + correction
        return out_flat.reshape(b, s, h, d).permute(0, 2, 1, 3)

    def forward_keys(self, keys: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self._lowrank_correct(k, self.A_k[i], self.B_k[i]) for i, k in enumerate(keys)]

    def forward_values(self, values: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self._lowrank_correct(v, self.A_v[i], self.B_v[i]) for i, v in enumerate(values)]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_cache(keys, values) -> DynamicCache:
    cache = DynamicCache()
    for k, v in zip(keys, values):
        cache.update(k, v, layer_idx=len(cache.layers))
    return cache


def zero_like(keys: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.zeros_like(k) for k in keys]


def roll_batch(keys: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.roll(k, shifts=1, dims=0) for k in keys]


def scored_logits(receiver, suffix_ids: torch.Tensor, cache: DynamicCache, target: torch.Tensor, batch: int) -> torch.Tensor:
    ids = torch.cat([suffix_ids.expand(batch, -1), target], dim=1)
    out = receiver(ids, past_key_values=cache, use_cache=False)
    return out.logits[:, -SCORED - 1 : -1, :]  # (batch, SCORED, vocab)


def batched_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))


def per_doc_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    logp = torch.log_softmax(logits.float(), dim=-1)  # (batch, SCORED, vocab)
    nll_per_tok = -logp.gather(2, target.unsqueeze(-1)).squeeze(-1)  # (batch, SCORED)
    correct_per_tok = (logits.argmax(-1) == target)  # (batch, SCORED)
    nll = nll_per_tok.mean(dim=1)  # (batch,)
    top1_exact = correct_per_tok.all(dim=1).float()
    top1_per_token = correct_per_tok.float().mean(dim=1)
    return {"nll": nll, "top1_exact": top1_exact, "top1_per_token": top1_per_token}


def bootstrap_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return [means[249], means[9749]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval().to(DEVICE)
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval().to(DEVICE)
    for m in (sender, receiver):
        for p in m.parameters():
            p.requires_grad_(False)

    eligible = two_token_codes(tokenizer, CANDIDATE_POOL_TARGET)
    train_codes = select_codes(eligible, "train", TRAIN_CODES)
    dev_codes = select_codes(eligible, "dev", DEV_CODES)
    assert set(train_codes).isdisjoint(dev_codes)
    suffix_ids = tokenizer.encode("The secret code is", return_tensors="pt").to(DEVICE)

    train_k, train_v, train_target = build_batch(sender, tokenizer, train_codes)
    dev_k, dev_v, dev_target = build_batch(sender, tokenizer, dev_codes)
    n_train = len(train_codes)
    n_dev = len(dev_codes)

    n_head = receiver.config.n_head
    head_dim = receiver.config.n_embd // n_head
    n_layers = len(SENDER_LAYER_SELECTION)

    adapter = LowRankKVAdapter(n_layers, head_dim, n_head, RANK, seed=SEED).to(DEVICE)
    opt = torch.optim.Adam(adapter.parameters(), lr=LR)

    k0 = adapter.forward_keys(train_k)
    v0 = adapter.forward_values(train_v)
    logits0 = scored_logits(receiver, suffix_ids, make_cache(k0, v0), train_target, n_train)
    loss0 = batched_ce(logits0, train_target)
    loss0.backward()
    b_grad_nonzero = any(adapter.B_k[i].grad.abs().max().item() > 0 for i in range(n_layers)) or \
                      any(adapter.B_v[i].grad.abs().max().item() > 0 for i in range(n_layers))
    if not b_grad_nonzero:
        raise RuntimeError("KV low-rank adapter B has no learning signal at init")
    opt.zero_grad()

    t0 = time.time()
    best_dev = math.inf
    best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
    dev_history: list[float] = []
    dev_top1_exact_history: list[float] = []
    dev_shuffled_top1_exact_history: list[float] = []
    dev_shuffled_minus_correct_history: list[float] = []
    dev_eval_epochs: list[int] = []
    train_loss_history: list[float] = []

    def dev_metrics() -> dict:
        adapter.eval()
        with torch.no_grad():
            k = adapter.forward_keys(dev_k)
            v = adapter.forward_values(dev_v)
            logits = scored_logits(receiver, suffix_ids, make_cache(k, v), dev_target, n_dev)
            m_correct = per_doc_metrics(logits, dev_target)
            loss = batched_ce(logits, dev_target).item()

            sk = roll_batch(dev_k)
            sv = roll_batch(dev_v)
            sk = adapter.forward_keys(sk)
            sv = adapter.forward_values(sv)
            slogits = scored_logits(receiver, suffix_ids, make_cache(sk, sv), dev_target, n_dev)
            m_shuf = per_doc_metrics(slogits, dev_target)
        adapter.train()
        return {
            "loss": loss,
            "top1_exact": m_correct["top1_exact"].mean().item(),
            "shuffled_top1_exact": m_shuf["top1_exact"].mean().item(),
            "shuffled_minus_correct": (m_shuf["nll"].mean() - m_correct["nll"].mean()).item(),
        }

    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    for epoch in range(MAX_EPOCHS):
        adapter.train()
        opt.zero_grad()
        k = adapter.forward_keys(train_k)
        v = adapter.forward_values(train_v)
        logits = scored_logits(receiver, suffix_ids, make_cache(k, v), train_target, n_train)
        loss = batched_ce(logits, train_target)
        loss.backward()
        opt.step()
        train_loss_history.append(loss.item())

        is_last = epoch == MAX_EPOCHS - 1
        if (epoch + 1) % DEV_EVAL_EVERY == 0 or is_last:
            m = dev_metrics()
            dl = m["loss"]
            dev_history.append(dl)
            dev_top1_exact_history.append(m["top1_exact"])
            dev_shuffled_top1_exact_history.append(m["shuffled_top1_exact"])
            dev_shuffled_minus_correct_history.append(m["shuffled_minus_correct"])
            dev_eval_epochs.append(epoch + 1)
            if dl < best_dev:
                best_dev = dl
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

        progress_path.write_text(json.dumps({
            "epochs_run": epoch + 1, "train_loss_history": train_loss_history,
            "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
            "dev_top1_exact_history": dev_top1_exact_history,
            "dev_shuffled_top1_exact_history": dev_shuffled_top1_exact_history,
            "dev_shuffled_minus_correct_history": dev_shuffled_minus_correct_history,
            "best_dev_loss_so_far": best_dev, "wall_seconds_so_far": time.time() - t0,
            "status": "training",
        }, indent=2) + "\n")

    adapter.load_state_dict(best_state)
    adapter.eval()
    wall_s = time.time() - t0

    window = train_loss_history[-(DEV_EVAL_EVERY + 1):]
    still_improving = False
    if len(window) >= 2 and window[0] != 0:
        rel_decrease = (window[0] - window[-1]) / abs(window[0])
        still_improving = rel_decrease > STILL_IMPROVING_REL_THRESHOLD

    with torch.no_grad():
        no_prefix_logits = scored_logits(receiver, suffix_ids, DynamicCache(), dev_target, n_dev)
        m_no_prefix = per_doc_metrics(no_prefix_logits, dev_target)

        zk = adapter.forward_keys(zero_like(dev_k))
        zv = adapter.forward_values(zero_like(dev_v))
        zero_logits = scored_logits(receiver, suffix_ids, make_cache(zk, zv), dev_target, n_dev)
        m_zero = per_doc_metrics(zero_logits, dev_target)

        ck = adapter.forward_keys(dev_k)
        cv = adapter.forward_values(dev_v)
        correct_logits = scored_logits(receiver, suffix_ids, make_cache(ck, cv), dev_target, n_dev)
        m_correct = per_doc_metrics(correct_logits, dev_target)

        sk = roll_batch(dev_k)
        sv = roll_batch(dev_v)
        sk = adapter.forward_keys(sk)
        sv = adapter.forward_values(sv)
        shuf_logits = scored_logits(receiver, suffix_ids, make_cache(sk, sv), dev_target, n_dev)
        m_shuf = per_doc_metrics(shuf_logits, dev_target)

    gains = (m_no_prefix["nll"] - m_correct["nll"]).tolist()
    content_advantage = (m_shuf["nll"] - m_correct["nll"]).tolist()
    zero_advantage = (m_zero["nll"] - m_correct["nll"]).tolist()
    finite = all(math.isfinite(x) for x in gains + content_advantage + zero_advantage)
    gain_ci = bootstrap_ci(gains, SEED)
    content_ci = bootstrap_ci(content_advantage, SEED + 1)
    zero_ci = bootstrap_ci(zero_advantage, SEED + 2)
    signal = finite and content_ci[0] > 0

    def summarize(m):
        return {"nll": m["nll"].mean().item(), "top1_exact": m["top1_exact"].mean().item(), "top1_per_token": m["top1_per_token"].mean().item()}

    result = {
        "seed": SEED, "sender_layer_selection": list(SENDER_LAYER_SELECTION), "rank": RANK, "scored_tokens": SCORED,
        "train_codes": train_codes, "dev_codes": dev_codes,
        "trainable_parameter_count": adapter.param_count(),
        "epochs_run": len(train_loss_history), "wall_seconds": wall_s,
        "best_dev_loss_at_selection": best_dev,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
        "dev_top1_exact_history": dev_top1_exact_history,
        "dev_shuffled_top1_exact_history": dev_shuffled_top1_exact_history,
        "dev_shuffled_minus_correct_history": dev_shuffled_minus_correct_history,
        "still_improving_at_cutoff": still_improving,
        "mechanism": "low_rank_residual_kv_adapter_batched_2token",
        "dev": {
            "no_prefix": summarize(m_no_prefix),
            "zero_sender": summarize(m_zero),
            "correct": summarize(m_correct),
            "shuffled_document": summarize(m_shuf),
            "no_prefix_minus_correct_nll": sum(gains) / len(gains),
            "no_prefix_minus_correct_bootstrap_95pct_ci": gain_ci,
            "shuffled_minus_correct_nll": sum(content_advantage) / len(content_advantage),
            "shuffled_minus_correct_bootstrap_95pct_ci": content_ci,
            "zero_sender_minus_correct_nll": sum(zero_advantage) / len(zero_advantage),
            "zero_sender_minus_correct_bootstrap_95pct_ci": zero_ci,
        },
        "finite": finite,
        "verdict": (
            "KV_ADAPTER_2TOKEN_TRAINING_INCONCLUSIVE" if still_improving else
            "KV_ADAPTER_2TOKEN_TRANSFER_SUPPORTED" if signal else
            "KV_ADAPTER_2TOKEN_NO_TRANSFER"
        ),
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(f"no_prefix: nll={result['dev']['no_prefix']['nll']:.4f} exact={result['dev']['no_prefix']['top1_exact']:.4f}")
    print(f"correct:   nll={result['dev']['correct']['nll']:.4f} exact={result['dev']['correct']['top1_exact']:.4f}")
    print(f"shuffled:  nll={result['dev']['shuffled_document']['nll']:.4f} exact={result['dev']['shuffled_document']['top1_exact']:.4f}")
    print(f"zero_sender: nll={result['dev']['zero_sender']['nll']:.4f} exact={result['dev']['zero_sender']['top1_exact']:.4f}")
    print(f"shuffled_minus_correct={result['dev']['shuffled_minus_correct_nll']:.4f} ci={content_ci} verdict={result['verdict']} wall_s={wall_s:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
