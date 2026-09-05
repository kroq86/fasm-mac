#!/usr/bin/env python3
"""Frozen-latent diagnostic probe: does the transferred latent contain
recoverable token1/token2 information that DistilGPT2's own decoding path
(9.38% / 0.00%) simply isn't extracting, or is the information genuinely
not there?

Everything upstream is frozen and untouched: reuses the exact bridge
weights already trained and saved by
phase1_controlled_fact_transfer_2token.py's diagnostic run
(phase1_controlled_fact_transfer_2token_diagnostic_weights.pt) and the
identical train/dev code split (same seed, same candidate-pool generation,
so codes match exactly). No further training of the bridge, sender, or
receiver. A separate, tiny linear classifier is trained per token position
on top of the frozen transferred latent (flattened across all 6 prefix
positions, 6*768=4608 features) to predict token1, and another to predict
token2 -- output classes restricted to the distinct token ids actually
observed across train+dev for that position (not the full ~50257 GPT-2
vocabulary), since fitting an unrestricted-vocabulary classifier from 64
examples would reintroduce the exact sample-complexity confound this probe
is trying to avoid conflating with a channel-capacity question.

Interpretation:
  - probe accuracy high (e.g. 70-80%+) while DistilGPT2's own decoding
    stays at ~9%/0% -> bottleneck is the receiver's decode/interface, not
    the channel: the information is there, DistilGPT2 just can't read it
    out through its own LM head in this configuration.
  - probe accuracy similarly low (~9%/0%) -> the transferred latent itself
    does not contain recoverable information at this fact length, under
    this training budget -- points toward data volume or genuine bridge
    capacity, not a decoder-interface problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 20260904
SENDER_LAYER = 12
TRAIN_CODES = 64
DEV_CODES = 32
CANDIDATE_SCAN_SEED = 424242
CANDIDATE_POOL_TARGET = 600
CODE_MIN, CODE_MAX = 100000, 999999
W_INIT_STD = 0.02
PROBE_EPOCHS = 300
PROBE_LR = 1e-2
PROBE_WEIGHT_DECAY = 1e-3


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


def prepare(sender, tokenizer, codes: list[int]) -> list[dict]:
    prepared = []
    with torch.no_grad():
        for code in codes:
            prefix_ids = tokenizer.encode(f"The secret code is {code}.", return_tensors="pt")
            target_ids = tokenizer.encode(f" {code}", add_special_tokens=False)
            if len(target_ids) != 2:
                raise RuntimeError(f"code {code} did not tokenize to exactly 2 tokens: {target_ids}")
            out = sender(prefix_ids, use_cache=False, output_hidden_states=True)
            H = out.hidden_states[SENDER_LAYER].float()
            H = torch.nn.functional.layer_norm(H, (H.shape[-1],)).detach()
            prepared.append({"code": code, "target_ids": target_ids, "H": H, "prefix_len": H.shape[1]})
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


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_probe(train_x: torch.Tensor, train_y: torch.Tensor, dev_x: torch.Tensor, dev_y: torch.Tensor, n_classes: int, seed: int) -> dict:
    torch.manual_seed(seed)
    probe = LinearProbe(train_x.shape[1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    best_dev_acc = -1.0
    best_dev_loss = math.inf
    history = []
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        opt.zero_grad()
        logits = probe(train_x)
        loss = torch.nn.functional.cross_entropy(logits, train_y)
        loss.backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            dev_logits = probe(dev_x)
            dev_loss = torch.nn.functional.cross_entropy(dev_logits, dev_y).item()
            dev_acc = (dev_logits.argmax(-1) == dev_y).float().mean().item()
            train_acc = (logits.argmax(-1) == train_y).float().mean().item()
        history.append({"epoch": epoch + 1, "train_loss": loss.item(), "train_acc": train_acc, "dev_loss": dev_loss, "dev_acc": dev_acc})
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_dev_acc = dev_acc

    return {
        "n_classes": n_classes,
        "final_train_acc": history[-1]["train_acc"],
        "final_dev_acc": history[-1]["dev_acc"],
        "best_dev_acc_at_best_dev_loss": best_dev_acc,
        "best_dev_loss": best_dev_loss,
        "history_first_last_5": history[:5] + history[-5:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--bridge-weights", type=Path, required=True)
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

    eligible = two_token_codes(tokenizer, CANDIDATE_POOL_TARGET)
    train_codes = select_codes(eligible, "train", TRAIN_CODES)
    dev_codes = select_codes(eligible, "dev", DEV_CODES)
    assert set(train_codes).isdisjoint(dev_codes)
    train = prepare(sender, tokenizer, train_codes)
    dev = prepare(sender, tokenizer, dev_codes)
    prefix_len = train[0]["prefix_len"]

    bridge = PositionwiseTranslator(768, target_norm=target_norm, seed=SEED)
    state = torch.load(args.bridge_weights)
    bridge.load_state_dict(state)
    bridge.eval()

    with torch.no_grad():
        train_latents = torch.stack([bridge(d["H"]).reshape(-1) for d in train])  # (n_train, prefix_len*768)
        dev_latents = torch.stack([bridge(d["H"]).reshape(-1) for d in dev])

    all_token1 = sorted({d["target_ids"][0] for d in train + dev})
    all_token2 = sorted({d["target_ids"][1] for d in train + dev})
    token1_to_class = {t: i for i, t in enumerate(all_token1)}
    token2_to_class = {t: i for i, t in enumerate(all_token2)}

    train_y1 = torch.tensor([token1_to_class[d["target_ids"][0]] for d in train], dtype=torch.long)
    dev_y1 = torch.tensor([token1_to_class[d["target_ids"][0]] for d in dev], dtype=torch.long)
    train_y2 = torch.tensor([token2_to_class[d["target_ids"][1]] for d in train], dtype=torch.long)
    dev_y2 = torch.tensor([token2_to_class[d["target_ids"][1]] for d in dev], dtype=torch.long)

    probe1 = train_probe(train_latents, train_y1, dev_latents, dev_y1, len(all_token1), SEED)
    probe2 = train_probe(train_latents, train_y2, dev_latents, dev_y2, len(all_token2), SEED + 1)

    # random-chance baselines for honest comparison
    chance1 = 1.0 / len(all_token1)
    chance2 = 1.0 / len(all_token2)

    result = {
        "seed": SEED,
        "bridge_weights_path": str(args.bridge_weights),
        "train_codes": train_codes, "dev_codes": dev_codes,
        "n_distinct_token1_classes": len(all_token1),
        "n_distinct_token2_classes": len(all_token2),
        "chance_accuracy_token1": chance1,
        "chance_accuracy_token2": chance2,
        "decoder_baseline_from_prior_run": {
            "token1_teacher_forced_and_autoregressive": 0.09375,
            "token2_given_true_or_generated_token1": 0.0,
        },
        "probe_token1": probe1,
        "probe_token2": probe2,
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(f"n_classes: token1={len(all_token1)} (chance={chance1:.4f}) token2={len(all_token2)} (chance={chance2:.4f})")
    print(f"probe token1: final_dev_acc={probe1['final_dev_acc']:.4f} best_dev_acc={probe1['best_dev_acc_at_best_dev_loss']:.4f}")
    print(f"probe token2: final_dev_acc={probe2['final_dev_acc']:.4f} best_dev_acc={probe2['best_dev_acc_at_best_dev_loss']:.4f}")
    print("decoder baseline: token1=0.0938 token2=0.0000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
