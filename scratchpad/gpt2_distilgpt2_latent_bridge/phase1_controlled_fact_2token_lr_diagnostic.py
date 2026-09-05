#!/usr/bin/env python3
"""Cheapest-possible falsification test for the optimization-regime
confound behind CONTROLLED_FACT_2TOKEN_TRAINING_INCONCLUSIVE.

Every prior 2-token controlled-fact run (64-train/300-epoch, 512-train/
40-epoch, and their diagnostic variants) used one fixed LR=1e-2 and never
reached a decisive top1_exact above 0% before hitting its epoch cap with
still_improving_at_cutoff=True. That confounds two different hypotheses:
(a) the channel cannot carry a 6-digit fact through this adapter at all,
or (b) LR=1e-2 is simply the wrong step size for a 768x768 matrix trained
one example at a time, and a different LR would converge inside a much
smaller epoch budget. Deciding between them changes how every downstream
negative verdict (addressing, compositionality) should be read, because
they all inherit the same fixed-LR training recipe.

This script changes exactly one variable at a time relative to the frozen
2-token setup: LR. Everything else -- 64 train / 32 dev codes (the size
that was already shown to produce a decisive result at the 1-token/3-digit
step, i.e. it is not the sample-complexity confound), same
PositionwiseTranslator architecture, same GPT-2 layer-12 site, same
norm-calibrated injection, same scored-window teacher-forcing metric,
same correct-vs-shuffled-document control -- is held fixed.

Training is a single full-batch gradient step per epoch over all 64 train
examples at once (one forward pass through the receiver on a stacked
(64, prefix_len, 768) tensor), not a 64-step-per-epoch per-example loop.
This matters for picking a fair epoch budget: every prior 2-token run
that used the per-example loop got ~64x more optimizer steps per epoch
than a full-batch step gives, so reusing a small epoch count under
full-batch would silently starve the optimizer rather than testing LR.
The one already-frozen precedent for this exact batching shape --
phase1_kv_cache_transfer_trained_batched.py, same TRAIN_CODES=64,
same full-batch-per-epoch design, same LR=1e-2 -- converged to
KV_ADAPTER_TRANSFER_SUPPORTED within 200 of its 800-epoch cap. 200 is
used here as the epoch cap for the same reason: it is the smallest
already-validated step budget for this exact batch shape, not a number
picked after seeing this script's own results.

Design (per user instruction, not tuned after seeing results):
  - 3 LRs x 1 seed x <=200 full-batch epochs (batch = all 64 train
    examples, matching the precedent's shape; not per-example SGD, not
    MPS, no seed sweep yet).
  - If all three LRs are equally dead (correct ~= shuffled, both ~0%
    top1_exact, shuffled_minus_correct CI still straddling zero) at their
    best dev checkpoint, optimization regime is ruled out as the
    explanation for the INCONCLUSIVE verdicts, and no further seeds are
    spent here.
  - If exactly one LR clearly separates correct from shuffled, this
    script prints that finding and stops; a human decision is required
    before spending the 2 extra confirmation seeds this LR would earn,
    per the preregistration discipline of not auto-chaining spikes.

This is a diagnostic, not a replacement for the frozen 2-token
preregistration gate: it can only shift which confound is live, not by
itself produce a CONTROLLED_FACT_2TOKEN_TRANSFER_SUPPORTED verdict.
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

SEED = 20260904
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SENDER_LAYER = 12
SCORED = 2
TRAIN_CODES = 64
DEV_CODES = 32
MAX_EPOCHS = 200
DEV_EVAL_EVERY = 10
STILL_IMPROVING_REL_THRESHOLD = 0.01
LR_GRID = [3e-3, 1e-2, 3e-2]
W_INIT_STD = 0.02
CODE_MIN, CODE_MAX = 100000, 999999
CANDIDATE_SCAN_SEED = 424242
CANDIDATE_POOL_TARGET = 400


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


def prepare(sender, tokenizer, codes: list[int]) -> dict:
    """Batched, not per-document: returns stacked tensors covering all
    codes at once, so training/eval run one forward pass per epoch."""
    hs, targets = [], []
    with torch.no_grad():
        for code in codes:
            prefix_ids = tokenizer.encode(f"The secret code is {code}.", return_tensors="pt").to(DEVICE)
            target_ids = tokenizer.encode(f" {code}", add_special_tokens=False)
            if len(target_ids) != SCORED:
                raise RuntimeError(f"code {code} did not tokenize to exactly {SCORED} tokens: {target_ids}")
            out = sender(prefix_ids, use_cache=False, output_hidden_states=True)
            H = out.hidden_states[SENDER_LAYER].float()
            H = torch.nn.functional.layer_norm(H, (H.shape[-1],))
            hs.append(H)  # (1, prefix_len, 768)
            targets.append(torch.tensor(target_ids, dtype=torch.long, device=DEVICE))
    prefix_lens = {h.shape[1] for h in hs}
    if len(prefix_lens) != 1:
        raise RuntimeError(f"prefix length not constant across documents: {prefix_lens}")
    return {
        "codes": codes,
        "H": torch.cat(hs, dim=0).detach(),          # (N, prefix_len, 768)
        "target": torch.stack(targets, dim=0),        # (N, SCORED)
        "prefix_len": prefix_lens.pop(),
    }


class PositionwiseTranslator(nn.Module):
    def __init__(self, width: int, target_norm: float, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(width, width, generator=g) * W_INIT_STD)
        self.target_norm = target_norm

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (N, prefix_len, width) -> (N, prefix_len, width), batched.
        raw = H @ self.W.T
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return raw / norm * self.target_norm

    def param_count(self) -> int:
        return self.W.numel()


def batched_logits(receiver, wte, suffix_ids: torch.Tensor, latent: torch.Tensor, target: torch.Tensor):
    """latent: (N, prefix_len, 768). target: (N, SCORED). Returns scored
    logits (N, SCORED, vocab) from one batched receiver forward pass."""
    n = latent.shape[0]
    receiver_ids = torch.cat([suffix_ids.unsqueeze(0).expand(n, -1), target], dim=1)
    receiver_embeds = wte(receiver_ids)
    combined = torch.cat([latent, receiver_embeds], dim=1)
    logits = receiver(inputs_embeds=combined, use_cache=False).logits
    return logits[:, -SCORED - 1 : -1, :]


def batched_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    """Per-document nll and exact-match, vectorized over the batch dim."""
    pred = logits.float()
    n = pred.shape[0]
    per_tok_loss = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none"
    ).reshape(n, SCORED)
    nll = per_tok_loss.mean(dim=1)                     # (N,)
    correct = (pred.argmax(-1) == target).all(dim=1)   # (N,)
    return {"nll": nll, "top1_exact": correct.float()}


def bootstrap_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return [means[249], means[9749]]


def eval_dev(bridge, dev: dict, receiver, wte, suffix_ids) -> dict:
    bridge.eval()
    with torch.no_grad():
        z = bridge(dev["H"])
        logits = batched_logits(receiver, wte, suffix_ids, z, dev["target"])
        correct = batched_metrics(logits, dev["target"])

        shuf_H = torch.roll(dev["H"], shifts=1, dims=0)
        z_shuf = bridge(shuf_H)
        logits_shuf = batched_logits(receiver, wte, suffix_ids, z_shuf, dev["target"])
        shuffled = batched_metrics(logits_shuf, dev["target"])
    bridge.train()
    content_advantage = (shuffled["nll"] - correct["nll"]).tolist()
    return {
        "correct_nll": correct["nll"].mean().item(),
        "shuffled_nll": shuffled["nll"].mean().item(),
        "correct_top1_exact": correct["top1_exact"].mean().item(),
        "shuffled_top1_exact": shuffled["top1_exact"].mean().item(),
        "shuffled_minus_correct_nll": (shuffled["nll"].mean() - correct["nll"].mean()).item(),
        "content_advantage_samples": content_advantage,
    }


def run_one_lr(lr: float, train: dict, dev: dict, receiver, wte, suffix_ids, target_norm) -> dict:
    bridge = PositionwiseTranslator(768, target_norm=target_norm, seed=SEED).to(DEVICE)
    opt = torch.optim.Adam(bridge.parameters(), lr=lr)

    t0 = time.time()
    best_dev_loss = math.inf
    best_state = {k: v.clone() for k, v in bridge.state_dict().items()}
    best_epoch = 0
    best_eval = None
    train_loss_history: list[float] = []
    dev_eval_epochs: list[int] = []
    dev_loss_history: list[float] = []
    dev_top1_history: list[float] = []

    for epoch in range(MAX_EPOCHS):
        bridge.train()
        opt.zero_grad()
        z = bridge(train["H"])
        logits = batched_logits(receiver, wte, suffix_ids, z, train["target"])
        loss = batched_metrics(logits, train["target"])["nll"].mean()
        loss.backward()
        opt.step()
        train_loss_history.append(loss.item())

        is_last = epoch == MAX_EPOCHS - 1
        if (epoch + 1) % DEV_EVAL_EVERY == 0 or is_last:
            ev = eval_dev(bridge, dev, receiver, wte, suffix_ids)
            dev_eval_epochs.append(epoch + 1)
            dev_loss_history.append(ev["correct_nll"])
            dev_top1_history.append(ev["correct_top1_exact"])
            if ev["correct_nll"] < best_dev_loss:
                best_dev_loss = ev["correct_nll"]
                best_state = {k: v.clone() for k, v in bridge.state_dict().items()}
                best_epoch = epoch + 1
                best_eval = ev

    window = train_loss_history[-(DEV_EVAL_EVERY + 1):]
    still_improving = False
    if len(window) >= 2 and window[0] != 0:
        rel_decrease = (window[0] - window[-1]) / abs(window[0])
        still_improving = rel_decrease > STILL_IMPROVING_REL_THRESHOLD

    ci = bootstrap_ci(best_eval["content_advantage_samples"], SEED + 1)
    wall_s = time.time() - t0
    return {
        "lr": lr, "wall_seconds": wall_s,
        "epochs_run": len(train_loss_history), "still_improving_at_cutoff": still_improving,
        "best_epoch": best_epoch, "best_dev_loss": best_dev_loss,
        "best_dev_correct_top1_exact": best_eval["correct_top1_exact"],
        "best_dev_shuffled_top1_exact": best_eval["shuffled_top1_exact"],
        "best_dev_shuffled_minus_correct_nll": best_eval["shuffled_minus_correct_nll"],
        "shuffled_minus_correct_bootstrap_95pct_ci": ci,
        "separates_from_shuffled": ci[0] > 0,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs,
        "dev_loss_history": dev_loss_history,
        "dev_top1_history": dev_top1_history,
    }


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
    wte = receiver.transformer.wte

    with torch.no_grad():
        target_norm = wte.weight.float().norm(dim=-1).median().item()

    eligible = two_token_codes(tokenizer, CANDIDATE_POOL_TARGET)
    train_codes = select_codes(eligible, "train", TRAIN_CODES)
    dev_codes = select_codes(eligible, "dev", DEV_CODES)
    assert set(train_codes).isdisjoint(dev_codes)
    train = prepare(sender, tokenizer, train_codes)
    dev = prepare(sender, tokenizer, dev_codes)
    assert train["prefix_len"] == dev["prefix_len"]

    suffix_ids = tokenizer.encode("The secret code is", return_tensors="pt")[0].to(DEVICE)

    runs = []
    for lr in LR_GRID:
        print(f"=== lr={lr} ===")
        r = run_one_lr(lr, train, dev, receiver, wte, suffix_ids, target_norm)
        runs.append(r)
        print(f"lr={lr}: best_epoch={r['best_epoch']} best_dev_loss={r['best_dev_loss']:.4f} "
              f"correct_top1={r['best_dev_correct_top1_exact']:.4f} shuffled_top1={r['best_dev_shuffled_top1_exact']:.4f} "
              f"shuffled_minus_correct={r['best_dev_shuffled_minus_correct_nll']:.4f} ci={r['shuffled_minus_correct_bootstrap_95pct_ci']} "
              f"still_improving={r['still_improving_at_cutoff']} wall_s={r['wall_seconds']:.1f}")

    separating = [r for r in runs if r["separates_from_shuffled"]]
    if not separating:
        verdict = "OPTIMIZATION_REGIME_RULED_OUT"
    elif len(separating) == 1:
        verdict = "OPTIMIZATION_REGIME_CANDIDATE_LR_FOUND"
    else:
        verdict = "OPTIMIZATION_REGIME_MULTIPLE_LR_SEPARATE"

    result = {
        "seed": SEED, "train_codes": train_codes, "dev_codes": dev_codes,
        "lr_grid": LR_GRID, "max_epochs": MAX_EPOCHS,
        "runs": runs, "verdict": verdict,
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(f"\nverdict={verdict}")
    if verdict == "OPTIMIZATION_REGIME_CANDIDATE_LR_FOUND":
        lr = separating[0]["lr"]
        print(f"lr={lr} separated correct from shuffled; per preregistration, do NOT auto-chain "
              f"into extra seeds -- confirm with a human before spending the 2 extra confirmation seeds.")
    elif verdict == "OPTIMIZATION_REGIME_RULED_OUT":
        print("all three LRs equally dead at their best dev checkpoint; optimization regime is "
              "not the explanation for CONTROLLED_FACT_2TOKEN_TRAINING_INCONCLUSIVE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
