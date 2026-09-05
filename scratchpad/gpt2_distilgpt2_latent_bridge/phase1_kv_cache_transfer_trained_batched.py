#!/usr/bin/env python3
"""Batched rewrite of phase1_kv_cache_transfer_trained.py -- identical
architecture, task, corpus, split, and gate; only the training-loop
mechanics change (one batched forward/backward per epoch instead of 64
separate batch=1 calls). Measured directly before writing this: batch=64
in one call is ~8.8x faster than 64 sequential batch=1 calls on MPS for
this exact model/shape (3.71s loop vs 0.42s batched, forward-pass only).
Not a new experiment -- a faster implementation of the same one, kept as
a separate file so the already-run per-document artifacts are untouched.

All documents share the same prefix length (verified single-BPE-token
3-digit codes -> constant tokenized length), so K/V tensors stack cleanly
along a new batch dimension. The low-rank adapter's per-layer (A,B) pair
is shared across the batch (that was always the design -- one adapter, not
one per document), so batching changes nothing about what is learned, only
how many forward passes it costs to compute the same gradient step.

Shuffled arm: batching the wrong-document pairing is a `torch.roll` by one
position along the batch dimension, exactly reproducing the `(i-1) % n`
pairing used in the per-document version.
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
CODE_MIN, CODE_MAX = 100, 999
SENDER_LAYER_SELECTION = (1, 3, 5, 7, 9, 11)
RANK = 8
A_INIT_STD = 0.02
MAX_EPOCHS = 800
DEV_EVAL_EVERY = 5
STILL_IMPROVING_REL_THRESHOLD = 0.01
LR = 1e-2


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bucket(code: int) -> str:
    value = int(sha(str(code).encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def single_token_codes(tokenizer) -> list[int]:
    return [c for c in range(CODE_MIN, CODE_MAX + 1) if len(tokenizer.encode(f" {c}", add_special_tokens=False)) == 1]


def select_codes(tokenizer, split: str, count: int) -> list[int]:
    eligible = single_token_codes(tokenizer)
    codes = sorted((c for c in eligible if bucket(c) == split), key=lambda c: sha(str(c).encode()))
    if len(codes) < count:
        raise RuntimeError(f"only {len(codes)} eligible {split} codes")
    return codes[:count]


def build_batch(sender, tokenizer, codes: list[int]):
    """Returns per-layer batched keys/values lists (each (batch,n_head,seq,head_dim))
    plus a target_ids tensor (batch,). Single batched sender forward pass
    (all prefixes share the same tokenized length by construction), not a
    per-document loop -- fixed after the 2-key script's per-document data
    prep turned out to dominate wall time despite the training loop itself
    being batched."""
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

    target_ids = []
    for code in codes:
        target_id = tokenizer.encode(f" {code}", add_special_tokens=False)
        assert len(target_id) == 1
        target_ids.append(target_id[0])
    target = torch.tensor(target_ids, dtype=torch.long, device=DEVICE)
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
        # x: (batch, n_head, seq, head_dim) -> (batch*seq, width) -> back
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


def batched_ce(receiver, suffix_ids: torch.Tensor, cache: DynamicCache, target: torch.Tensor, batch: int) -> torch.Tensor:
    ids = suffix_ids.expand(batch, -1)
    out = receiver(ids, past_key_values=cache, use_cache=False)
    logits = out.logits[:, -1, :].float()
    return torch.nn.functional.cross_entropy(logits, target)


def per_doc_nll_top1(receiver, suffix_ids: torch.Tensor, cache: DynamicCache, target: torch.Tensor, batch: int) -> tuple[list[float], list[float]]:
    ids = suffix_ids.expand(batch, -1)
    with torch.no_grad():
        out = receiver(ids, past_key_values=cache, use_cache=False)
    logits = out.logits[:, -1, :].float()
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    top1 = (logits.argmax(-1) == target).float()
    return nll.tolist(), top1.tolist()


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

    train_codes = select_codes(tokenizer, "train", TRAIN_CODES)
    dev_codes = select_codes(tokenizer, "dev", DEV_CODES)
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
    loss0 = batched_ce(receiver, suffix_ids, make_cache(k0, v0), train_target, n_train)
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
    dev_top1_history: list[float] = []
    dev_shuffled_top1_history: list[float] = []
    dev_shuffled_minus_correct_history: list[float] = []
    dev_eval_epochs: list[int] = []
    train_loss_history: list[float] = []

    def dev_metrics() -> dict:
        """Correct-arm and shuffled-arm dev loss/top1, tracked every eval
        point -- not just loss -- so stability of the alignment (not just
        its existence at one snapshot) can be inspected across steps."""
        adapter.eval()
        with torch.no_grad():
            k = adapter.forward_keys(dev_k)
            v = adapter.forward_values(dev_v)
            correct_loss = batched_ce(receiver, suffix_ids, make_cache(k, v), dev_target, n_dev).item()
            correct_nll, correct_top1 = per_doc_nll_top1(receiver, suffix_ids, make_cache(k, v), dev_target, n_dev)

            sk = roll_batch(dev_k)
            sv = roll_batch(dev_v)
            sk = adapter.forward_keys(sk)
            sv = adapter.forward_values(sv)
            shuf_nll, shuf_top1 = per_doc_nll_top1(receiver, suffix_ids, make_cache(sk, sv), dev_target, n_dev)
        adapter.train()
        return {
            "loss": correct_loss,
            "top1": sum(correct_top1) / len(correct_top1),
            "shuffled_top1": sum(shuf_top1) / len(shuf_top1),
            "shuffled_minus_correct": sum(shuf_nll) / len(shuf_nll) - sum(correct_nll) / len(correct_nll),
        }

    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    for epoch in range(MAX_EPOCHS):
        adapter.train()
        opt.zero_grad()
        k = adapter.forward_keys(train_k)
        v = adapter.forward_values(train_v)
        loss = batched_ce(receiver, suffix_ids, make_cache(k, v), train_target, n_train)
        loss.backward()
        opt.step()
        train_loss_history.append(loss.item())

        is_last = epoch == MAX_EPOCHS - 1
        if (epoch + 1) % DEV_EVAL_EVERY == 0 or is_last:
            m = dev_metrics()
            dl = m["loss"]
            dev_history.append(dl)
            dev_top1_history.append(m["top1"])
            dev_shuffled_top1_history.append(m["shuffled_top1"])
            dev_shuffled_minus_correct_history.append(m["shuffled_minus_correct"])
            dev_eval_epochs.append(epoch + 1)
            if dl < best_dev:
                best_dev = dl
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

        progress_path.write_text(json.dumps({
            "epochs_run": epoch + 1, "train_loss_history": train_loss_history,
            "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
            "dev_top1_history": dev_top1_history,
            "dev_shuffled_top1_history": dev_shuffled_top1_history,
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

    no_prefix_cache = DynamicCache()
    no_prefix_nll, no_prefix_top1 = per_doc_nll_top1(receiver, suffix_ids, no_prefix_cache, dev_target, n_dev)

    zk = adapter.forward_keys(zero_like(dev_k))
    zv = adapter.forward_values(zero_like(dev_v))
    zero_nll, zero_top1 = per_doc_nll_top1(receiver, suffix_ids, make_cache(zk, zv), dev_target, n_dev)

    ck = adapter.forward_keys(dev_k)
    cv = adapter.forward_values(dev_v)
    correct_nll, correct_top1 = per_doc_nll_top1(receiver, suffix_ids, make_cache(ck, cv), dev_target, n_dev)

    sk = roll_batch(dev_k)
    sv = roll_batch(dev_v)
    sk = adapter.forward_keys(sk)
    sv = adapter.forward_values(sv)
    shuf_nll, shuf_top1 = per_doc_nll_top1(receiver, suffix_ids, make_cache(sk, sv), dev_target, n_dev)

    gains = [a - c for a, c in zip(no_prefix_nll, correct_nll)]
    content_advantage = [s - c for s, c in zip(shuf_nll, correct_nll)]
    zero_advantage = [z - c for z, c in zip(zero_nll, correct_nll)]
    finite = all(math.isfinite(x) for rows in (no_prefix_nll, zero_nll, correct_nll, shuf_nll) for x in rows)
    gain_ci = bootstrap_ci(gains, SEED)
    content_ci = bootstrap_ci(content_advantage, SEED + 1)
    zero_ci = bootstrap_ci(zero_advantage, SEED + 2)
    signal = finite and content_ci[0] > 0

    def summarize(nll_list, top1_list):
        return {"nll": sum(nll_list) / len(nll_list), "top1": sum(top1_list) / len(top1_list)}

    result = {
        "seed": SEED, "sender_layer_selection": list(SENDER_LAYER_SELECTION), "rank": RANK,
        "trainable_parameter_count": adapter.param_count(),
        "epochs_run": len(train_loss_history), "wall_seconds": wall_s,
        "best_dev_loss_at_selection": best_dev,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
        "dev_top1_history": dev_top1_history,
        "dev_shuffled_top1_history": dev_shuffled_top1_history,
        "dev_shuffled_minus_correct_history": dev_shuffled_minus_correct_history,
        "still_improving_at_cutoff": still_improving,
        "mechanism": "low_rank_residual_kv_adapter_batched",
        "dev": {
            "no_prefix": summarize(no_prefix_nll, no_prefix_top1),
            "zero_sender": summarize(zero_nll, zero_top1),
            "correct": summarize(correct_nll, correct_top1),
            "shuffled_document": summarize(shuf_nll, shuf_top1),
            "no_prefix_minus_correct_nll": sum(gains) / len(gains),
            "no_prefix_minus_correct_bootstrap_95pct_ci": gain_ci,
            "shuffled_minus_correct_nll": sum(content_advantage) / len(content_advantage),
            "shuffled_minus_correct_bootstrap_95pct_ci": content_ci,
            "zero_sender_minus_correct_nll": sum(zero_advantage) / len(zero_advantage),
            "zero_sender_minus_correct_bootstrap_95pct_ci": zero_ci,
        },
        "finite": finite,
        "verdict": (
            "KV_ADAPTER_TRAINING_INCONCLUSIVE" if still_improving else
            "KV_ADAPTER_TRANSFER_SUPPORTED" if signal else
            "KV_ADAPTER_NO_TRANSFER"
        ),
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(f"no_prefix: nll={result['dev']['no_prefix']['nll']:.4f} top1={result['dev']['no_prefix']['top1']:.4f}")
    print(f"correct:   nll={result['dev']['correct']['nll']:.4f} top1={result['dev']['correct']['top1']:.4f}")
    print(f"shuffled:  nll={result['dev']['shuffled_document']['nll']:.4f} top1={result['dev']['shuffled_document']['top1']:.4f}")
    print(f"zero_sender: nll={result['dev']['zero_sender']['nll']:.4f} top1={result['dev']['zero_sender']['top1']:.4f}")
    print(f"shuffled_minus_correct={result['dev']['shuffled_minus_correct_nll']:.4f} ci={content_ci} verdict={result['verdict']} wall_s={wall_s:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
