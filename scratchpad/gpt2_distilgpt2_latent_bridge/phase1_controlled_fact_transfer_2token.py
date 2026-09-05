#!/usr/bin/env python3
"""Step 2 of the controlled-fact ladder: does the channel carry a 2-token
(6-digit) fact, not just a 1-token (3-digit) one?

Step 1 (phase1_controlled_fact_transfer.py, 3-digit/1-token codes) gave a
decisive positive: correct top-1 = 81.25% vs 0% for every control,
shuffled_minus_correct = 18.354 nats (CI [16.059, 20.542]), converged
smoothly over 300 epochs with no overfitting cliff. This step changes
exactly one variable: fact length/bandwidth. Same architecture
(PositionwiseTranslator, no bias, norm-calibrated, prepended to input
embeddings), same site (GPT-2 final layer), same sha256-bucket train/dev
split discipline (disjoint by construction), same constant receiver-side
suffix text. Only the fact itself grows from 1 token to 2 (a 6-digit
number that tokenizes to exactly 2 BPE tokens -- verified per-candidate,
not assumed uniform; ~21% of random 6-digit numbers qualify).

Scoring now uses the standard scored-window teacher-forcing convention
used throughout this project (phase1_scalar_sweep.py etc): the full
receiver sequence (suffix + both code tokens) is fed, and
`logits[:, -SCORED-1:-1]` against `target = ids[:, -SCORED:]` scores both
code-token predictions with real-token teacher forcing between them.
top1_exact requires BOTH tokens correct (whole-fact recall); per-token
top1 is also recorded for diagnostic granularity.

Claim: correct-vs-shuffled gap remains large and CI stays far from zero.
Falsified by: the effect collapsing toward the ~0.005-nat scale seen on
natural continuation, which would indicate bandwidth degrades sharply
past a single token rather than the channel being broadly capable.
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
SENDER_LAYER = 12
SCORED = 2
TRAIN_CODES = 512  # sample-complexity check: identical architecture/site/
# objective/gate as the 64-example run; only training-set size changed, to
# separate "not enough data for this fact length's larger label space"
# from "genuine bridge/channel capacity ceiling" -- the frozen-latent
# probe already ruled out a decoder/interface bottleneck (probe matched
# the decoder's 9.38%/0.00% almost exactly).
DEV_CODES = 32
MAX_EPOCHS = 40  # 512-train sample-complexity run: 300 epochs was
# calibrated for 64 examples/epoch (19,200 gradient steps); at 512
# examples/epoch, 40 epochs gives a comparable ~20,000 steps without an
# 8x-longer wall clock (each epoch costs ~8x more at this train size).
DEV_EVAL_EVERY = 2
STILL_IMPROVING_REL_THRESHOLD = 0.01
LR = 1e-2
W_INIT_STD = 0.02
CODE_MIN, CODE_MAX = 100000, 999999
CANDIDATE_SCAN_SEED = 424242
CANDIDATE_POOL_TARGET = 1600  # comfortably more than TRAIN_CODES+DEV_CODES
# at the ~70/15/15 train/dev/test bucket split (512 train needs >=~730
# eligible train-bucket codes at 70% hit rate; scanning a larger pool
# supplies that with margin)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bucket(code: int) -> str:
    value = int(sha(str(code).encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def two_token_codes(tokenizer, pool_target: int) -> list[int]:
    # Empirically verified: only ~21% of random 6-digit numbers tokenize
    # to exactly 2 BPE tokens (checked directly, not assumed). Sample
    # deterministically until enough eligible codes are found.
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


def prepare(sender, tokenizer, codes: list[int]) -> list[dict]:
    prepared = []
    with torch.no_grad():
        for code in codes:
            prefix_ids = tokenizer.encode(f"The secret code is {code}.", return_tensors="pt")
            target_ids = tokenizer.encode(f" {code}", add_special_tokens=False)
            if len(target_ids) != SCORED:
                raise RuntimeError(f"code {code} did not tokenize to exactly {SCORED} tokens: {target_ids}")
            out = sender(prefix_ids, use_cache=False, output_hidden_states=True)
            H = out.hidden_states[SENDER_LAYER].float()
            H = torch.nn.functional.layer_norm(H, (H.shape[-1],)).detach()
            prepared.append({
                "code": code, "target_ids": torch.tensor(target_ids, dtype=torch.long),
                "H": H, "prefix_len": H.shape[1],
            })
    return prepared


class PositionwiseTranslator(nn.Module):
    def __init__(self, width: int, target_norm: float, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(width, width, generator=g) * W_INIT_STD)
        self.target_norm = target_norm

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        raw = H.squeeze(0) @ self.W.T
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return raw / norm * self.target_norm

    def param_count(self) -> int:
        return self.W.numel()


def receiver_scored_logits(receiver, wte, receiver_ids: torch.Tensor, latent: torch.Tensor | None):
    receiver_embeds = wte(receiver_ids)
    combined = receiver_embeds if latent is None else torch.cat([latent.unsqueeze(0), receiver_embeds], dim=1)
    logits = receiver(inputs_embeds=combined, use_cache=False).logits
    return logits[:, -SCORED - 1 : -1, :]  # (1, SCORED, vocab)


def scored_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = logits.float()
    losses = torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none")
    return losses.mean()


def per_doc_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    """logits/target here are already teacher-forced: both scored
    positions use the REAL preceding token as context (receiver_ids feeds
    the true target_ids in full), not the model's own generated guess.
    token2's accuracy below is therefore token2-given-true-token1, not
    token2-given-generated-token1 -- see autoregressive_metrics for the
    genuinely different (not yet previously measured) comparison."""
    pred = logits.float()
    losses = torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none")
    pred0 = pred[0]      # (SCORED, vocab)
    tgt0 = target[0]     # (SCORED,)
    per_token_correct = (pred0.argmax(-1) == tgt0)  # (SCORED,)
    logp = torch.log_softmax(pred0, dim=-1)
    ranks = []
    for i in range(pred0.shape[0]):
        sorted_ids = pred0[i].argsort(descending=True)
        rank = (sorted_ids == tgt0[i]).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)
    return {
        "nll": losses.mean().item(),
        "top1_exact": float(per_token_correct.all().item()),
        "top1_per_token": per_token_correct.float().mean().item(),
        "top1_token1": float(per_token_correct[0].item()),
        "top1_token2_given_true_token1": float(per_token_correct[1].item()),
        "logprob_true_token1": logp[0, tgt0[0]].item(),
        "logprob_true_token2_given_true_token1": logp[1, tgt0[1]].item(),
        "rank_token1": ranks[0],
        "rank_token2_given_true_token1": ranks[1],
    }


def autoregressive_metrics(receiver, wte, suffix_ids: torch.Tensor, latent: torch.Tensor | None, target: torch.Tensor) -> dict:
    """Genuine autoregressive generation, not teacher-forced: token1 is
    greedily decoded from the model's own distribution, then THAT
    (possibly wrong) token1 is fed back in to predict token2 -- the
    comparison point that actually isolates autoregressive error
    propagation from channel capacity."""
    suffix_embeds = wte(suffix_ids).unsqueeze(0)
    combined = suffix_embeds if latent is None else torch.cat([latent.unsqueeze(0), suffix_embeds], dim=1)
    logits1 = receiver(inputs_embeds=combined, use_cache=False).logits[:, -1, :]
    gen_token1 = logits1.argmax(-1)
    top1_token1 = float((gen_token1 == target[0, 0]).item())

    gen1_embed = wte(gen_token1).unsqueeze(0)
    combined2 = torch.cat([combined, gen1_embed], dim=1)
    logits2 = receiver(inputs_embeds=combined2, use_cache=False).logits[:, -1, :]
    gen_token2 = logits2.argmax(-1)
    top1_token2_given_generated_token1 = float((gen_token2 == target[0, 1]).item())

    return {
        "top1_token1": top1_token1,
        "top1_token2_given_generated_token1": top1_token2_given_generated_token1,
        "top1_exact_autoregressive": float(top1_token1 == 1.0 and top1_token2_given_generated_token1 == 1.0),
    }


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
    ap.add_argument("--bridge-out", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval()
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
    prefix_lens = {d["prefix_len"] for d in train + dev}
    assert len(prefix_lens) == 1, f"prefix length not constant across documents: {prefix_lens}"
    prefix_len = prefix_lens.pop()

    suffix_ids = tokenizer.encode("The secret code is", return_tensors="pt")[0]

    def receiver_ids_for(target_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([suffix_ids, target_ids]).unsqueeze(0)

    bridge = PositionwiseTranslator(768, target_norm=target_norm, seed=SEED)
    opt = torch.optim.Adam(bridge.parameters(), lr=LR)

    z0 = bridge(train[0]["H"])
    rid0 = receiver_ids_for(train[0]["target_ids"])
    logits0 = receiver_scored_logits(receiver, wte, rid0, z0)
    loss0 = scored_ce(logits0, train[0]["target_ids"].unsqueeze(0))
    loss0.backward()
    g = bridge.W.grad
    if g is None or not torch.isfinite(g).all() or g.abs().max().item() == 0.0:
        raise RuntimeError("2-token controlled-fact translator initialization has no learning signal")
    opt.zero_grad()

    t0 = time.time()
    best_dev = math.inf
    best_state = {k: v.clone() for k, v in bridge.state_dict().items()}
    dev_history: list[float] = []
    dev_eval_epochs: list[int] = []
    train_loss_history: list[float] = []

    def dev_loss() -> float:
        bridge.eval()
        with torch.no_grad():
            total = 0.0
            for d in dev:
                z = bridge(d["H"])
                rid = receiver_ids_for(d["target_ids"])
                logits = receiver_scored_logits(receiver, wte, rid, z)
                total += scored_ce(logits, d["target_ids"].unsqueeze(0)).item()
        bridge.train()
        return total / len(dev)

    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    for epoch in range(MAX_EPOCHS):
        bridge.train()
        epoch_losses = []
        for d in train:
            opt.zero_grad()
            z = bridge(d["H"])
            rid = receiver_ids_for(d["target_ids"])
            logits = receiver_scored_logits(receiver, wte, rid, z)
            loss = scored_ce(logits, d["target_ids"].unsqueeze(0))
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
                best_state = {k: v.clone() for k, v in bridge.state_dict().items()}

        progress_path.write_text(json.dumps({
            "epochs_run": epoch + 1, "train_loss_history": train_loss_history,
            "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
            "best_dev_loss_so_far": best_dev, "wall_seconds_so_far": time.time() - t0,
            "status": "training",
        }, indent=2) + "\n")

    bridge.load_state_dict(best_state)
    bridge.eval()
    wall_s = time.time() - t0

    window = train_loss_history[-(DEV_EVAL_EVERY + 1):]
    still_improving = False
    if len(window) >= 2 and window[0] != 0:
        rel_decrease = (window[0] - window[-1]) / abs(window[0])
        still_improving = rel_decrease > STILL_IMPROVING_REL_THRESHOLD

    torch.save(bridge.state_dict(), args.bridge_out)
    bridge_sha256 = sha(json.dumps(
        {k: v.tolist() for k, v in best_state.items()}, sort_keys=True
    ).encode())

    n = len(dev)
    no_prefix_rows, zero_sender_rows, chosen_rows, shuffled_rows = [], [], [], []
    chosen_autoregressive_rows, shuffled_autoregressive_rows = [], []
    with torch.no_grad():
        for i, d in enumerate(dev):
            rid = receiver_ids_for(d["target_ids"])
            tgt = d["target_ids"].unsqueeze(0)

            logits = receiver_scored_logits(receiver, wte, rid, None)
            no_prefix_rows.append({"code": d["code"], **per_doc_metrics(logits, tgt)})

            zero_latent = torch.zeros(prefix_len, 768)
            logits = receiver_scored_logits(receiver, wte, rid, zero_latent)
            zero_sender_rows.append({"code": d["code"], **per_doc_metrics(logits, tgt)})

            z = bridge(d["H"])
            logits = receiver_scored_logits(receiver, wte, rid, z)
            chosen_rows.append({"code": d["code"], **per_doc_metrics(logits, tgt)})
            chosen_autoregressive_rows.append({"code": d["code"], **autoregressive_metrics(receiver, wte, suffix_ids, z, tgt)})

            other_H = dev[(i - 1) % n]["H"]
            z_shuf = bridge(other_H)
            logits = receiver_scored_logits(receiver, wte, rid, z_shuf)
            shuffled_rows.append({"code": d["code"], **per_doc_metrics(logits, tgt)})
            shuffled_autoregressive_rows.append({"code": d["code"], **autoregressive_metrics(receiver, wte, suffix_ids, z_shuf, tgt)})

    gains = [a["nll"] - c["nll"] for a, c in zip(no_prefix_rows, chosen_rows)]
    content_advantage = [s["nll"] - c["nll"] for s, c in zip(shuffled_rows, chosen_rows)]
    zero_advantage = [z["nll"] - c["nll"] for z, c in zip(zero_sender_rows, chosen_rows)]
    finite = all(
        math.isfinite(r["nll"]) for rows in (no_prefix_rows, zero_sender_rows, chosen_rows, shuffled_rows) for r in rows
    )
    gain_ci = bootstrap_ci(gains, SEED)
    content_ci = bootstrap_ci(content_advantage, SEED + 1)
    zero_ci = bootstrap_ci(zero_advantage, SEED + 2)
    signal = finite and content_ci[0] > 0

    def summarize(rows):
        return {
            "nll": sum(r["nll"] for r in rows) / n,
            "top1_exact": sum(r["top1_exact"] for r in rows) / n,
            "top1_per_token": sum(r["top1_per_token"] for r in rows) / n,
            "top1_token1": sum(r["top1_token1"] for r in rows) / n,
            "top1_token2_given_true_token1": sum(r["top1_token2_given_true_token1"] for r in rows) / n,
            "mean_rank_token1": sum(r["rank_token1"] for r in rows) / n,
            "mean_rank_token2_given_true_token1": sum(r["rank_token2_given_true_token1"] for r in rows) / n,
            "mean_logprob_true_token1": sum(r["logprob_true_token1"] for r in rows) / n,
            "mean_logprob_true_token2_given_true_token1": sum(r["logprob_true_token2_given_true_token1"] for r in rows) / n,
        }

    def summarize_autoregressive(rows):
        return {
            "top1_token1": sum(r["top1_token1"] for r in rows) / n,
            "top1_token2_given_generated_token1": sum(r["top1_token2_given_generated_token1"] for r in rows) / n,
            "top1_exact_autoregressive": sum(r["top1_exact_autoregressive"] for r in rows) / n,
        }

    result = {
        "seed": SEED, "sender_layer": SENDER_LAYER, "scored_tokens": SCORED,
        "train_codes": train_codes, "dev_codes": dev_codes,
        "prefix_len": prefix_len, "target_norm": target_norm,
        "bridge": "positionwise_translator_controlled_fact_2token_task",
        "trainable_parameter_count": bridge.param_count(),
        "bridge_sha256": bridge_sha256,
        "epochs_run": len(train_loss_history), "wall_seconds": wall_s,
        "best_dev_loss_at_selection": best_dev,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
        "still_improving_at_cutoff": still_improving,
        "dev": {
            "no_prefix": summarize(no_prefix_rows),
            "zero_sender": summarize(zero_sender_rows),
            "correct": summarize(chosen_rows),
            "shuffled_document": summarize(shuffled_rows),
            "correct_autoregressive": summarize_autoregressive(chosen_autoregressive_rows),
            "shuffled_autoregressive": summarize_autoregressive(shuffled_autoregressive_rows),
            "no_prefix_minus_correct_nll": sum(gains) / len(gains),
            "no_prefix_minus_correct_bootstrap_95pct_ci": gain_ci,
            "shuffled_minus_correct_nll": sum(content_advantage) / len(content_advantage),
            "shuffled_minus_correct_bootstrap_95pct_ci": content_ci,
            "zero_sender_minus_correct_nll": sum(zero_advantage) / len(zero_advantage),
            "zero_sender_minus_correct_bootstrap_95pct_ci": zero_ci,
        },
        "finite": finite,
        "verdict": (
            "CONTROLLED_FACT_2TOKEN_TRAINING_INCONCLUSIVE" if still_improving else
            "CONTROLLED_FACT_2TOKEN_TRANSFER_SUPPORTED" if signal else
            "CONTROLLED_FACT_2TOKEN_NO_TRANSFER"
        ),
        "per_document_dev": {
            "no_prefix": no_prefix_rows, "zero_sender": zero_sender_rows,
            "correct": chosen_rows, "shuffled_document": shuffled_rows,
            "correct_autoregressive": chosen_autoregressive_rows,
            "shuffled_autoregressive": shuffled_autoregressive_rows,
        },
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(f"no_prefix: nll={result['dev']['no_prefix']['nll']:.4f} exact={result['dev']['no_prefix']['top1_exact']:.4f}")
    print(f"correct teacher-forced: token1={result['dev']['correct']['top1_token1']:.4f} "
          f"token2|true_token1={result['dev']['correct']['top1_token2_given_true_token1']:.4f} "
          f"rank1={result['dev']['correct']['mean_rank_token1']:.2f} rank2={result['dev']['correct']['mean_rank_token2_given_true_token1']:.2f}")
    print(f"correct autoregressive: token1={result['dev']['correct_autoregressive']['top1_token1']:.4f} "
          f"token2|generated_token1={result['dev']['correct_autoregressive']['top1_token2_given_generated_token1']:.4f} "
          f"exact={result['dev']['correct_autoregressive']['top1_exact_autoregressive']:.4f}")
    print(f"correct:   nll={result['dev']['correct']['nll']:.4f} exact={result['dev']['correct']['top1_exact']:.4f}")
    print(f"shuffled:  nll={result['dev']['shuffled_document']['nll']:.4f} exact={result['dev']['shuffled_document']['top1_exact']:.4f}")
    print(f"shuffled_minus_correct={result['dev']['shuffled_minus_correct_nll']:.4f} ci={content_ci} verdict={result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
