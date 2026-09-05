#!/usr/bin/env python3
"""No-training sensitivity probe for a frozen GPT-2 -> DistilGPT2 bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TEXTS = (
    "The quick brown fox jumps over the lazy dog.",
    "A small compiler transforms a graph into an execution plan.",
    "The temperature increased after the pump stopped.",
    "Machine learning models predict tokens from context.",
    "A transaction commits only after every invariant passes.",
    "The astronomer counted bright stars in the northern sky.",
    "Memory bandwidth often limits autoregressive decoding.",
    "The bridge carries information between two frozen models.",
    "If the battery is low, start the backup generator.",
    "Numerical verification compares outputs within a tolerance.",
    "The cat waited quietly beside the open window.",
    "Software tests should fail when required evidence is missing.",
    "Rain fell overnight and the river rose by morning.",
    "An attention head combines values selected by query and key.",
    "The engineer measured latency before changing the kernel.",
    "A reliable system records intent before performing a side effect.",
)


def corpus_sha() -> str:
    return hashlib.sha256(("\n".join(TEXTS) + "\n").encode()).hexdigest()


def nll(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    pred = logits[:, :-1].float()
    target = ids[:, 1:]
    valid = mask[:, 1:].bool()
    losses = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none"
    ).reshape_as(target)
    mean = losses[valid].mean().item()
    top1 = (pred.argmax(-1)[valid] == target[valid]).float().mean().item()
    return mean, top1


def injected_forward(model, block_index: int, replacement: torch.Tensor, ids, mask):
    block = model.transformer.h[block_index]

    def replace(_module, args):
        return (replacement,) + args[1:]

    handle = block.register_forward_pre_hook(replace)
    try:
        return model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    finally:
        handle.remove()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    torch.manual_seed(20260904)
    random.seed(20260904)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(list(TEXTS), return_tensors="pt", padding=True)
    ids, mask = batch["input_ids"], batch["attention_mask"]

    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval()
    for model in (sender, receiver):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    with torch.inference_mode():
        sout = sender(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
        rout = receiver(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
        # Frozen boundary: sender after block 5 -> receiver before block 3.
        aligned = sout.hidden_states[6]
        wrong_layer = sout.hidden_states[1]
        shuffled_example = aligned.roll(1, 0)
        shuffled_position = aligned.roll(1, 1)
        permutation = torch.randperm(aligned.shape[-1])
        signs = torch.where(torch.rand(aligned.shape[-1]) < 0.5, -1.0, 1.0)
        random_isometry = aligned[..., permutation] * signs

        arms = {
            "receiver_alone": rout.logits,
            "sender_alone": sout.logits,
            "identity_aligned": injected_forward(receiver, 3, aligned, ids, mask),
            "shuffled_example": injected_forward(receiver, 3, shuffled_example, ids, mask),
            "shuffled_position": injected_forward(receiver, 3, shuffled_position, ids, mask),
            "wrong_sender_layer": injected_forward(receiver, 3, wrong_layer, ids, mask),
            "random_isometry": injected_forward(receiver, 3, random_isometry, ids, mask),
        }

    metrics = {}
    for name, logits in arms.items():
        loss, accuracy = nll(logits, ids, mask)
        if not math.isfinite(loss):
            raise RuntimeError(f"non-finite NLL in {name}")
        metrics[name] = {"nll": loss, "perplexity": math.exp(loss), "top1": accuracy}
    base = arms["identity_aligned"]
    for name, logits in arms.items():
        metrics[name]["max_abs_vs_identity_logits"] = (logits.float() - base.float()).abs().max().item()

    receiver_nll = metrics["receiver_alone"]["nll"]
    sender_nll = metrics["sender_alone"]["nll"]
    identity_nll = metrics["identity_aligned"]["nll"]
    gap = receiver_nll - sender_nll
    gap_closed = (receiver_nll - identity_nll) / gap if gap > 0 else float("nan")
    corrupt_controls = ("shuffled_example", "shuffled_position", "wrong_sender_layer", "random_isometry")
    instrument_sensitive = all(identity_nll + 0.1 < metrics[name]["nll"] for name in corrupt_controls)

    result = {
        "seed": 20260904,
        "corpus_sha256": corpus_sha(),
        "examples": len(TEXTS),
        "tokens_scored": int(mask[:, 1:].sum()),
        "sender_boundary": "hidden_states[6] (after GPT-2 block 5)",
        "receiver_boundary": "before DistilGPT2 block 3",
        "receiver_to_sender_nll_gap_closed_by_identity": gap_closed,
        "instrument_sensitive": instrument_sensitive,
        "metrics": metrics,
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    return 0 if instrument_sensitive else 1


if __name__ == "__main__":
    raise SystemExit(main())
