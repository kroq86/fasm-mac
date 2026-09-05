#!/usr/bin/env python3
"""Trained low-rank K/V adapter on top of the raw KV-cache transplant.

phase1_kv_cache_transfer.py (raw, zero trainable parameters) already found
real signal: shuffled_minus_correct = 1.038 nats, CI [0.811, 1.288],
excludes zero -- but top-1 exact recovery was 0%. This adds the smallest
plausible correction, not a dense adapter (which would repeat the
dense-no-bias/positionwise-translator overfitting failure mode on only 64
training examples): a per-layer, per-K/V low-rank residual,
`X' = X + (X @ A) @ B`, A:(64,r), B:(r,64), r=8, no bias, applied
independently at each of the 6 selected layers to K and to V (12 adapters
total, ~147K params combined). B is initialized to exact zero, so at
init the adapter is the identity -- output equals the already-informative
raw baseline exactly, not a random, unaligned transform (avoiding the
catastrophic-destabilization failure the embedding-prefix rung hit before
norm calibration was added). A is small-random (matches the project's
established safe-init convention for this class of bilinear parameterization).

With K=V=0 (zero sender), the correction term is also 0 by construction
(it multiplies K), so the zero-sender arm remains architecturally forced
to reduce to the null baseline, same discipline as every no-bias rung in
this project.

Same task, corpus, split, and primary gate as phase1_kv_cache_transfer.py.
Trains on next-token cross-entropy for the correct code, evaluated on the
same four arms (no_prefix, zero_sender, correct, shuffled).
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
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # ~4.6x faster
# than CPU even for this small a model at batch=1, measured directly
# before switching (53.5ms/call CPU vs 11.6ms/call MPS)
TRAIN_CODES = 64
DEV_CODES = 32
CODE_MIN, CODE_MAX = 100, 999
SENDER_LAYER_SELECTION = (1, 3, 5, 7, 9, 11)
RANK = 8
A_INIT_STD = 0.02
MAX_EPOCHS = 800  # convergence-completion: 300-epoch run gave the largest
# effect in this project (shuffled_minus_correct=21.23 nats, top1=34.4%)
# but was still_improving_at_cutoff -- cheap to extend on MPS (~2.6s/epoch)
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


def sender_cache(sender, tokenizer, code: int):
    ids = tokenizer.encode(f"The secret code is {code}.", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = sender(ids, use_cache=True)
    pkv = out.past_key_values
    keys = [pkv.layers[i].keys.clone().detach() for i in SENDER_LAYER_SELECTION]
    values = [pkv.layers[i].values.clone().detach() for i in SENDER_LAYER_SELECTION]
    return keys, values


class LowRankKVAdapter(nn.Module):
    """One (A,B) pair per selected layer per K/V stream. head_dim=64.
    X: (1, n_head=12, seq, 64) -> flatten heads*dim=768 per position ->
    residual low-rank correction -> reshape back."""

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
        # x: (1, n_head, seq, head_dim) -> (seq, width)
        b, h, s, d = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(s, h * d)
        correction = (flat @ A) @ B
        out_flat = flat + correction
        return out_flat.reshape(1, s, h, d).permute(0, 2, 1, 3)

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


def zero_like_keys(keys: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.zeros_like(k) for k in keys]


def scored_ce(receiver, suffix_ids: torch.Tensor, cache: DynamicCache, target_id: int) -> torch.Tensor:
    out = receiver(suffix_ids, past_key_values=cache, use_cache=False)
    logits = out.logits[:, -1, :].float()
    return torch.nn.functional.cross_entropy(logits, torch.tensor([target_id], device=logits.device))


def nll_top1_from_logits(logits: torch.Tensor, target_id: int) -> tuple[float, float]:
    logp = torch.log_softmax(logits.float(), dim=-1)[0]
    nll = -logp[target_id].item()
    top1 = float(logits.argmax(-1).item() == target_id)
    return nll, top1


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

    def build(codes):
        rows = []
        for code in codes:
            keys, values = sender_cache(sender, tokenizer, code)
            target_id = tokenizer.encode(f" {code}", add_special_tokens=False)
            assert len(target_id) == 1
            rows.append({"code": code, "keys": keys, "values": values, "target_id": target_id[0]})
        return rows

    train = build(train_codes)
    dev = build(dev_codes)
    n_head = receiver.config.n_head
    head_dim = receiver.config.n_embd // n_head
    n_layers = len(SENDER_LAYER_SELECTION)

    adapter = LowRankKVAdapter(n_layers, head_dim, n_head, RANK, seed=SEED).to(DEVICE)
    opt = torch.optim.Adam(adapter.parameters(), lr=LR)

    d0 = train[0]
    k0 = adapter.forward_keys(d0["keys"])
    v0 = adapter.forward_values(d0["values"])
    loss0 = scored_ce(receiver, suffix_ids, make_cache(k0, v0), d0["target_id"])
    loss0.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    if not grads_ok:
        raise RuntimeError("KV low-rank adapter initialization has non-finite/missing gradient")
    # B is zero at init, so dL/dA is generically zero too (correction=(flat@A)@B, dCorrection/dA involves B);
    # only dL/dB is expected nonzero at this exact init -- check that explicitly instead of requiring all grads nonzero.
    b_grad_nonzero = any(adapter.B_k[i].grad.abs().max().item() > 0 for i in range(n_layers)) or \
                      any(adapter.B_v[i].grad.abs().max().item() > 0 for i in range(n_layers))
    if not b_grad_nonzero:
        raise RuntimeError("KV low-rank adapter B has no learning signal at init")
    opt.zero_grad()

    t0 = time.time()
    best_dev = math.inf
    best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
    dev_history: list[float] = []
    dev_eval_epochs: list[int] = []
    train_loss_history: list[float] = []

    def dev_loss() -> float:
        adapter.eval()
        total = 0.0
        with torch.no_grad():
            for d in dev:
                k = adapter.forward_keys(d["keys"])
                v = adapter.forward_values(d["values"])
                total += scored_ce(receiver, suffix_ids, make_cache(k, v), d["target_id"]).item()
        adapter.train()
        return total / len(dev)

    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    for epoch in range(MAX_EPOCHS):
        adapter.train()
        epoch_losses = []
        for d in train:
            opt.zero_grad()
            k = adapter.forward_keys(d["keys"])
            v = adapter.forward_values(d["values"])
            loss = scored_ce(receiver, suffix_ids, make_cache(k, v), d["target_id"])
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        train_loss_history.append(sum(epoch_losses) / len(epoch_losses))

        is_last = epoch == MAX_EPOCHS - 1
        if (epoch + 1) % DEV_EVAL_EVERY == 0 or is_last:
            dl = dev_loss()
            dev_history.append(dl)
            dev_eval_epochs.append(epoch + 1)
            if dl < best_dev:
                best_dev = dl
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

        progress_path.write_text(json.dumps({
            "epochs_run": epoch + 1, "train_loss_history": train_loss_history,
            "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
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

    n = len(dev)
    no_prefix_rows, zero_sender_rows, chosen_rows, shuffled_rows = [], [], [], []
    with torch.no_grad():
        for i, d in enumerate(dev):
            out = receiver(suffix_ids, past_key_values=DynamicCache(), use_cache=False)
            nll, top1 = nll_top1_from_logits(out.logits[:, -1, :], d["target_id"])
            no_prefix_rows.append({"code": d["code"], "nll": nll, "top1": top1})

            zk = adapter.forward_keys(zero_like_keys(d["keys"]))
            zv = adapter.forward_values(zero_like_keys(d["values"]))
            out = receiver(suffix_ids, past_key_values=make_cache(zk, zv), use_cache=False)
            nll, top1 = nll_top1_from_logits(out.logits[:, -1, :], d["target_id"])
            zero_sender_rows.append({"code": d["code"], "nll": nll, "top1": top1})

            k = adapter.forward_keys(d["keys"])
            v = adapter.forward_values(d["values"])
            out = receiver(suffix_ids, past_key_values=make_cache(k, v), use_cache=False)
            nll, top1 = nll_top1_from_logits(out.logits[:, -1, :], d["target_id"])
            chosen_rows.append({"code": d["code"], "nll": nll, "top1": top1})

            other = dev[(i - 1) % n]
            k = adapter.forward_keys(other["keys"])
            v = adapter.forward_values(other["values"])
            out = receiver(suffix_ids, past_key_values=make_cache(k, v), use_cache=False)
            nll, top1 = nll_top1_from_logits(out.logits[:, -1, :], d["target_id"])
            shuffled_rows.append({"code": d["code"], "nll": nll, "top1": top1})

    gains = [a["nll"] - c["nll"] for a, c in zip(no_prefix_rows, chosen_rows)]
    content_advantage = [s["nll"] - c["nll"] for s, c in zip(shuffled_rows, chosen_rows)]
    zero_advantage = [z["nll"] - c["nll"] for z, c in zip(zero_sender_rows, chosen_rows)]
    finite = all(math.isfinite(r["nll"]) for rows in (no_prefix_rows, zero_sender_rows, chosen_rows, shuffled_rows) for r in rows)
    gain_ci = bootstrap_ci(gains, SEED)
    content_ci = bootstrap_ci(content_advantage, SEED + 1)
    zero_ci = bootstrap_ci(zero_advantage, SEED + 2)
    signal = finite and content_ci[0] > 0

    def summarize(rows):
        return {"nll": sum(r["nll"] for r in rows) / n, "top1": sum(r["top1"] for r in rows) / n}

    result = {
        "seed": SEED, "sender_layer_selection": list(SENDER_LAYER_SELECTION), "rank": RANK,
        "trainable_parameter_count": adapter.param_count(),
        "epochs_run": len(train_loss_history), "wall_seconds": wall_s,
        "best_dev_loss_at_selection": best_dev,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
        "still_improving_at_cutoff": still_improving,
        "mechanism": "low_rank_residual_kv_adapter",
        "dev": {
            "no_prefix": summarize(no_prefix_rows),
            "zero_sender": summarize(zero_sender_rows),
            "correct": summarize(chosen_rows),
            "shuffled_document": summarize(shuffled_rows),
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
        "per_document_dev": {
            "no_prefix": no_prefix_rows, "zero_sender": zero_sender_rows,
            "correct": chosen_rows, "shuffled_document": shuffled_rows,
        },
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(f"no_prefix: nll={result['dev']['no_prefix']['nll']:.4f} top1={result['dev']['no_prefix']['top1']:.4f}")
    print(f"correct:   nll={result['dev']['correct']['nll']:.4f} top1={result['dev']['correct']['top1']:.4f}")
    print(f"shuffled:  nll={result['dev']['shuffled_document']['nll']:.4f} top1={result['dev']['shuffled_document']['top1']:.4f}")
    print(f"zero_sender: nll={result['dev']['zero_sender']['nll']:.4f} top1={result['dev']['zero_sender']['top1']:.4f}")
    print(f"shuffled_minus_correct={result['dev']['shuffled_minus_correct_nll']:.4f} ci={content_ci} verdict={result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
