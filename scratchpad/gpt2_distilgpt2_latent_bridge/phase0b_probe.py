#!/usr/bin/env python3
"""Withheld-prefix, no-training latent-interface probe.

Sender sees prefix+suffix only for the information-ceiling arm.  The injected
arm receives a sender state computed from prefix alone; the receiver never sees
the prefix tokens.  Test-bucket documents are fingerprinted but never scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 20260904
PREFIX = 64
SUFFIX = 160
SCORED = 32
DEV_DOCS = 32


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_documents(path: Path) -> list[str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    docs = [" ".join(part.split()) for part in text.split("<|endoftext|>")]
    docs = [doc for doc in docs if doc]
    if len(docs) != len(set(docs)):
        raise RuntimeError("exact duplicate documents in source corpus")
    return docs


def bucket(doc: str) -> str:
    value = int(sha(doc.encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def score_last(logits: torch.Tensor, ids: torch.Tensor) -> tuple[float, float]:
    pred = logits[:, -SCORED - 1 : -1].float()
    target = ids[:, -SCORED:]
    losses = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none"
    ).reshape(pred.shape[0], SCORED)
    top1 = (pred.argmax(-1) == target).float()
    return losses.mean().item(), top1.mean().item()


def bootstrap_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return [means[249], means[9749]]


def injected_forward(
    model, block_index: int, replacement: torch.Tensor, ids: torch.Tensor, row: int = 0
):
    block = model.transformer.h[block_index]

    def replace(_module, args):
        hidden = args[0].clone()
        hidden[:, row, :] = replacement
        return (hidden,) + args[1:]

    handle = block.register_forward_pre_hook(replace)
    try:
        return model(input_ids=ids, use_cache=False).logits
    finally:
        handle.remove()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval()
    for model in (sender, receiver):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    docs = load_documents(args.corpus)
    split_hashes = {
        split: sha(("\n".join(sorted(sha(d.encode()) for d in docs if bucket(d) == split)) + "\n").encode())
        for split in ("train", "dev", "test")
    }

    selected: list[tuple[str, list[int]]] = []
    for doc in sorted((d for d in docs if bucket(d) == "dev"), key=lambda d: sha(d.encode())):
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if len(ids) >= PREFIX + SUFFIX:
            selected.append((sha(doc.encode()), ids[: PREFIX + SUFFIX]))
        if len(selected) == DEV_DOCS:
            break
    if len(selected) != DEV_DOCS:
        raise RuntimeError(f"only {len(selected)} eligible dev documents")

    per_doc = []
    with torch.inference_mode():
        for index, (doc_sha, tokens) in enumerate(selected):
            prefix = torch.tensor([tokens[:PREFIX]], dtype=torch.long)
            suffix = torch.tensor([tokens[PREFIX:]], dtype=torch.long)
            receiver_ids = torch.cat((torch.tensor([[tokenizer.eos_token_id]]), suffix), dim=1)
            full_ids = torch.tensor([tokens], dtype=torch.long)

            prefix_out = sender(prefix, use_cache=False, output_hidden_states=True)
            aligned = prefix_out.hidden_states[6][:, -1, :]
            wrong_layer = prefix_out.hidden_states[1][:, -1, :]
            random_state = aligned[..., torch.randperm(aligned.shape[-1])]

            receiver_logits = receiver(receiver_ids, use_cache=False).logits
            full_logits = sender(full_ids, use_cache=False).logits
            identity_logits = injected_forward(receiver, 3, aligned, receiver_ids)
            wrong_position_logits = injected_forward(receiver, 3, aligned, receiver_ids, row=64)
            wrong_layer_logits = injected_forward(receiver, 3, wrong_layer, receiver_ids)
            random_logits = injected_forward(receiver, 3, random_state, receiver_ids)

            # Deterministic cross-document control: state from the previous
            # selected document is evaluated in a second pass below.
            per_doc.append({
                "index": index,
                "doc_sha256": doc_sha,
                "receiver": score_last(receiver_logits, receiver_ids),
                "full_sender": score_last(full_logits, full_ids),
                "identity": score_last(identity_logits, receiver_ids),
                "wrong_position": score_last(wrong_position_logits, receiver_ids),
                "wrong_layer": score_last(wrong_layer_logits, receiver_ids),
                "random_permutation": score_last(random_logits, receiver_ids),
                "aligned": aligned.cpu(),
                "receiver_ids": receiver_ids,
            })

        for index, row in enumerate(per_doc):
            foreign = per_doc[(index - 1) % len(per_doc)]["aligned"]
            logits = injected_forward(receiver, 3, foreign, row["receiver_ids"])
            row["shuffled_document"] = score_last(logits, row["receiver_ids"])

    arm_names = (
        "receiver", "full_sender", "identity", "wrong_position", "wrong_layer",
        "random_permutation", "shuffled_document",
    )
    metrics = {}
    for arm in arm_names:
        nlls = [float(row[arm][0]) for row in per_doc]
        top1s = [float(row[arm][1]) for row in per_doc]
        metrics[arm] = {"nll": sum(nlls) / len(nlls), "top1": sum(top1s) / len(top1s)}
    for row in per_doc:
        del row["aligned"]
        del row["receiver_ids"]
        for arm in arm_names:
            row[arm] = {"nll": row[arm][0], "top1": row[arm][1]}

    headroom = metrics["receiver"]["nll"] - metrics["full_sender"]["nll"]
    identity_gain = metrics["receiver"]["nll"] - metrics["identity"]["nll"]
    gap_closed = identity_gain / headroom if headroom > 0 else float("nan")
    paired = {
        "headroom_receiver_minus_full_sender": [
            row["receiver"]["nll"] - row["full_sender"]["nll"] for row in per_doc
        ],
        "identity_gain_receiver_minus_identity": [
            row["receiver"]["nll"] - row["identity"]["nll"] for row in per_doc
        ],
    }
    for control in ("wrong_position", "wrong_layer", "random_permutation", "shuffled_document"):
        paired[f"identity_advantage_{control}_minus_identity"] = [
            row[control]["nll"] - row["identity"]["nll"] for row in per_doc
        ]
    paired_summary = {
        name: {
            "mean": sum(values) / len(values),
            "bootstrap_95pct_ci": bootstrap_ci(values, SEED + i),
        }
        for i, (name, values) in enumerate(paired.items())
    }
    finite = all(math.isfinite(v) for arm in metrics.values() for v in arm.values())
    headroom_present = paired_summary["headroom_receiver_minus_full_sender"]["bootstrap_95pct_ci"][0] > 0.15
    identity_ci = paired_summary["identity_gain_receiver_minus_identity"]["bootstrap_95pct_ci"]
    identity_transfers = identity_ci[0] > 0
    verdict = (
        "INCONCLUSIVE_NO_HEADROOM" if not headroom_present else
        "IDENTITY_TRANSFERS_WITHHELD_INFORMATION" if identity_transfers else
        "HEADROOM_PRESENT_IDENTITY_DOES_NOT_TRANSFER"
    )
    result = {
        "seed": SEED,
        "corpus_path_basename": args.corpus.name,
        "corpus_sha256": sha(args.corpus.read_bytes()),
        "split_document_hashes_sha256": split_hashes,
        "test_scored": False,
        "prefix_tokens": PREFIX,
        "suffix_tokens": SUFFIX,
        "scored_suffix_tokens": SCORED,
        "dev_documents": len(selected),
        "headroom_nll": headroom,
        "identity_nll_gain": identity_gain,
        "identity_fraction_of_headroom": gap_closed,
        "paired_document_bootstrap": paired_summary,
        "finite": finite,
        "verdict": verdict,
        "metrics": metrics,
        "per_document": per_doc,
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    return 0 if finite and headroom_present else 2


if __name__ == "__main__":
    raise SystemExit(main())
