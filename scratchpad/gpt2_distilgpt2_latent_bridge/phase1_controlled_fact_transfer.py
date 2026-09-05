#!/usr/bin/env python3
"""Separate preregistered instrument: can the channel transfer even a
single controlled, discrete fact -- not "does natural-continuation NLL
move by a few thousandths of a nat".

Claim: TinyStories may be too noisy a measurement instrument. DistilGPT2
can often continue a natural suffix plausibly with no information at all,
so a real content-specific signal could be swamped by measurement noise,
which would explain the tiny (~0.005 nat) effects seen throughout this
investigation without meaning the channel carries no information. This
test removes that ambiguity: GPT-2 (sender) sees "The secret code is
{code}." (a single 3-digit number, tokenizes to exactly one token with a
leading space, verified directly against the tokenizer); DistilGPT2
(receiver) sees only the constant, code-independent text "The secret code
is" and must predict the next token. If a working bridge exists at all,
correct should give receiver near-certainty on the right code while
shuffled gives near-certainty on a *different* (wrong) code -- a large,
unambiguous NLL gap, not a borderline one. Falsified by: correct still
indistinguishable from shuffled even on this single-fact task -- which
would mean the problem is more fundamental than TinyStories' measurement
noise, i.e. this architecture/site/model-pair genuinely cannot carry even
one discrete fact.

Same architecture as phase1_full_passthrough_trained_translator.py
(PositionwiseTranslator: shared per-position linear map, no bias, applied
to all 6 prefix-sentence positions independently, then norm-calibrated to
the receiver's real median embedding norm, then prepended to the
receiver's input embeddings) -- only the corpus/task changes, not the
bridge family or fusion mechanism. Codes are 3-digit numbers 100-999
(single BPE token each, verified), split into disjoint train/dev/test sets
by the same sha256-bucket convention used throughout this project (bucket
on the code's own hash, not a document's), so the projector must learn a
genuine transcoding function rather than memorize a train-set lookup
table. Suffix text is IDENTICAL across every document by construction
(only the injected latent varies) -- this removes any possible confound
from text content itself. Primary gate, unchanged: correct-vs-shuffled
paired bootstrap CI > 0 on this single next-token prediction.
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
TRAIN_CODES = 64
DEV_CODES = 32
MAX_EPOCHS = 300  # convergence-completion rerun: the 30-epoch result showed
# a massive (~5.7 nat) correct-vs-shuffled gap with dev still improving at
# cutoff (best checkpoint was the last epoch, not an early overfitting
# peak) -- unlike prior rungs, this task trains in seconds, so extending
# the budget to let it actually converge is cheap and well-justified, not
# a tuning search. Same architecture/data/loss/gate as the 30-epoch run.
DEV_EVAL_EVERY = 5
STILL_IMPROVING_REL_THRESHOLD = 0.01
LR = 1e-2
W_INIT_STD = 0.02
CODE_MIN, CODE_MAX = 100, 999  # inclusive; all tokenize to exactly one BPE token


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bucket(code: int) -> str:
    value = int(sha(str(code).encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def single_token_codes(tokenizer) -> list[int]:
    # empirically verified: not every 3-digit number is one BPE token
    # (e.g. 886 -> [' 88', '6']); filter rather than assume the whole range
    return [
        c for c in range(CODE_MIN, CODE_MAX + 1)
        if len(tokenizer.encode(f" {c}", add_special_tokens=False)) == 1
    ]


def select_codes(tokenizer, split: str, count: int) -> list[int]:
    eligible = single_token_codes(tokenizer)
    codes = sorted((c for c in eligible if bucket(c) == split), key=lambda c: sha(str(c).encode()))
    if len(codes) < count:
        raise RuntimeError(f"only {len(codes)} eligible {split} codes")
    return codes[:count]


def prepare(sender, tokenizer, codes: list[int]) -> list[dict]:
    prepared = []
    with torch.no_grad():
        for code in codes:
            prefix_ids = tokenizer.encode(f"The secret code is {code}.", return_tensors="pt")
            target_id = tokenizer.encode(f" {code}", add_special_tokens=False)
            if len(target_id) != 1:
                raise RuntimeError(f"code {code} did not tokenize to exactly one token: {target_id}")
            out = sender(prefix_ids, use_cache=False, output_hidden_states=True)
            H = out.hidden_states[SENDER_LAYER].float()  # (1, prefix_len, 768)
            H = torch.nn.functional.layer_norm(H, (H.shape[-1],)).detach()
            prepared.append({"code": code, "target_id": target_id[0], "H": H, "prefix_len": H.shape[1]})
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


def receiver_next_token_logits(receiver, wte, suffix_ids: torch.Tensor, latent: torch.Tensor | None):
    suffix_embeds = wte(suffix_ids)
    combined = suffix_embeds if latent is None else torch.cat([latent.unsqueeze(0), suffix_embeds], dim=1)
    logits = receiver(inputs_embeds=combined, use_cache=False).logits
    return logits[:, -1, :]  # next-token prediction after the full fed sequence


def nll_top1(logits_last: torch.Tensor, target_id: int) -> tuple[float, float]:
    logp = torch.log_softmax(logits_last.float(), dim=-1)[0]
    nll = -logp[target_id].item()
    top1 = float(logits_last.argmax(-1).item() == target_id)
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
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval()
    for m in (sender, receiver):
        for p in m.parameters():
            p.requires_grad_(False)
    wte = receiver.transformer.wte

    with torch.no_grad():
        target_norm = wte.weight.float().norm(dim=-1).median().item()

    train_codes = select_codes(tokenizer, "train", TRAIN_CODES)
    dev_codes = select_codes(tokenizer, "dev", DEV_CODES)
    assert set(train_codes).isdisjoint(dev_codes)
    train = prepare(sender, tokenizer, train_codes)
    dev = prepare(sender, tokenizer, dev_codes)
    prefix_lens = {d["prefix_len"] for d in train + dev}
    assert len(prefix_lens) == 1, f"prefix length not constant across documents: {prefix_lens}"
    prefix_len = prefix_lens.pop()

    suffix_ids = tokenizer.encode("The secret code is", return_tensors="pt")

    bridge = PositionwiseTranslator(768, target_norm=target_norm, seed=SEED)
    opt = torch.optim.Adam(bridge.parameters(), lr=LR)

    z0 = bridge(train[0]["H"])
    logits0 = receiver_next_token_logits(receiver, wte, suffix_ids, z0)
    loss0 = -torch.log_softmax(logits0.float(), dim=-1)[0, train[0]["target_id"]]
    loss0.backward()
    g = bridge.W.grad
    if g is None or not torch.isfinite(g).all() or g.abs().max().item() == 0.0:
        raise RuntimeError("controlled-fact translator initialization has no learning signal")
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
                logits = receiver_next_token_logits(receiver, wte, suffix_ids, z)
                nll, _ = nll_top1(logits, d["target_id"])
                total += nll
        bridge.train()
        return total / len(dev)

    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    for epoch in range(MAX_EPOCHS):
        bridge.train()
        epoch_losses = []
        for d in train:
            opt.zero_grad()
            z = bridge(d["H"])
            logits = receiver_next_token_logits(receiver, wte, suffix_ids, z)
            loss = -torch.log_softmax(logits.float(), dim=-1)[0, d["target_id"]]
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

    n = len(dev)
    no_prefix_rows, zero_sender_rows, chosen_rows, shuffled_rows = [], [], [], []
    with torch.no_grad():
        for i, d in enumerate(dev):
            logits = receiver_next_token_logits(receiver, wte, suffix_ids, None)
            nll, top1 = nll_top1(logits, d["target_id"])
            no_prefix_rows.append({"code": d["code"], "nll": nll, "top1": top1})

            zero_latent = torch.zeros(prefix_len, 768)
            logits = receiver_next_token_logits(receiver, wte, suffix_ids, zero_latent)
            nll, top1 = nll_top1(logits, d["target_id"])
            zero_sender_rows.append({"code": d["code"], "nll": nll, "top1": top1})

            z = bridge(d["H"])
            logits = receiver_next_token_logits(receiver, wte, suffix_ids, z)
            nll, top1 = nll_top1(logits, d["target_id"])
            chosen_rows.append({"code": d["code"], "nll": nll, "top1": top1})

            other_H = dev[(i - 1) % n]["H"]
            z_shuf = bridge(other_H)
            logits = receiver_next_token_logits(receiver, wte, suffix_ids, z_shuf)
            nll, top1 = nll_top1(logits, d["target_id"])
            shuffled_rows.append({"code": d["code"], "nll": nll, "top1": top1})

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

    result = {
        "seed": SEED, "sender_layer": SENDER_LAYER,
        "train_codes": train_codes, "dev_codes": dev_codes,
        "prefix_len": prefix_len, "target_norm": target_norm,
        "bridge": "positionwise_translator_controlled_fact_task",
        "trainable_parameter_count": bridge.param_count(),
        "epochs_run": len(train_loss_history), "wall_seconds": wall_s,
        "best_dev_loss_at_selection": best_dev,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
        "still_improving_at_cutoff": still_improving,
        "dev": {
            "no_prefix": {"nll": sum(r["nll"] for r in no_prefix_rows) / n, "top1": sum(r["top1"] for r in no_prefix_rows) / n},
            "zero_sender": {"nll": sum(r["nll"] for r in zero_sender_rows) / n, "top1": sum(r["top1"] for r in zero_sender_rows) / n},
            "correct": {"nll": sum(r["nll"] for r in chosen_rows) / n, "top1": sum(r["top1"] for r in chosen_rows) / n},
            "shuffled_document": {"nll": sum(r["nll"] for r in shuffled_rows) / n, "top1": sum(r["top1"] for r in shuffled_rows) / n},
            "no_prefix_minus_correct_nll": sum(gains) / len(gains),
            "no_prefix_minus_correct_bootstrap_95pct_ci": gain_ci,
            "shuffled_minus_correct_nll": sum(content_advantage) / len(content_advantage),
            "shuffled_minus_correct_bootstrap_95pct_ci": content_ci,
            "zero_sender_minus_correct_nll": sum(zero_advantage) / len(zero_advantage),
            "zero_sender_minus_correct_bootstrap_95pct_ci": zero_ci,
        },
        "finite": finite,
        "verdict": (
            "CONTROLLED_FACT_TRAINING_INCONCLUSIVE" if still_improving else
            "CONTROLLED_FACT_TRANSFER_SUPPORTED" if signal else
            "CONTROLLED_FACT_NO_TRANSFER"
        ),
        "per_document_dev": {
            "no_prefix": no_prefix_rows, "zero_sender": zero_sender_rows,
            "correct": chosen_rows, "shuffled_document": shuffled_rows,
        },
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(f"no_prefix_nll={result['dev']['no_prefix']['nll']:.4f} top1={result['dev']['no_prefix']['top1']:.4f}")
    print(f"correct_nll={result['dev']['correct']['nll']:.4f} top1={result['dev']['correct']['top1']:.4f}")
    print(f"shuffled_nll={result['dev']['shuffled_document']['nll']:.4f} top1={result['dev']['shuffled_document']['top1']:.4f}")
    print(f"shuffled_minus_correct={result['dev']['shuffled_minus_correct_nll']:.4f} ci={content_ci} verdict={result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
