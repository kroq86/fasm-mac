#!/usr/bin/env python3
"""First minimum-alignment rung: LayerNorm(sender state) times one scalar.

The scalar is selected on the train bucket from a frozen grid.  Dev is scored
once after selection.  Test-bucket documents are fingerprinted but never
tokenized or scored.  Both language models remain frozen.
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
TRAIN_DOCS = 64
DEV_DOCS = 32
ALPHAS = (
    -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, -0.125,
    0.0,
    0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_documents(path: Path) -> list[str]:
    docs = [" ".join(x.split()) for x in path.read_text().split("<|endoftext|>")]
    docs = [x for x in docs if x]
    if len(docs) != len(set(docs)):
        raise RuntimeError("exact duplicate documents in source corpus")
    return docs


def bucket(doc: str) -> str:
    value = int(sha(doc.encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def select(tokenizer, docs: list[str], split: str, count: int) -> list[dict]:
    chosen = []
    for doc in sorted((x for x in docs if bucket(x) == split), key=lambda x: sha(x.encode())):
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if len(ids) >= PREFIX + SUFFIX:
            chosen.append({"sha256": sha(doc.encode()), "ids": ids[: PREFIX + SUFFIX]})
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"only {len(chosen)} eligible {split} documents")
    return chosen


def prepare(sender, tokenizer, records: list[dict]) -> list[dict]:
    prepared = []
    with torch.inference_mode():
        for record in records:
            tokens = record["ids"]
            prefix = torch.tensor([tokens[:PREFIX]], dtype=torch.long)
            suffix = torch.tensor([tokens[PREFIX:]], dtype=torch.long)
            receiver_ids = torch.cat((torch.tensor([[tokenizer.eos_token_id]]), suffix), dim=1)
            out = sender(prefix, use_cache=False, output_hidden_states=True)
            h = out.hidden_states[6][:, -1, :].float()
            h = torch.nn.functional.layer_norm(h, (h.shape[-1],))
            prepared.append({
                "sha256": record["sha256"], "ids": receiver_ids,
                "full_ids": torch.tensor([tokens], dtype=torch.long), "h": h,
            })
    return prepared


def run_injected(receiver, prepared: list[dict], alpha: float, shuffled: bool = False):
    block = receiver.transformer.h[3]
    state = {"extra": None}

    def hook(_module, args):
        hidden = args[0].clone()
        hidden[:, 0, :] += state["extra"]
        return (hidden,) + args[1:]

    handle = block.register_forward_pre_hook(hook)
    rows = []
    try:
        with torch.inference_mode():
            for i, record in enumerate(prepared):
                source = prepared[(i - 1) % len(prepared)] if shuffled else record
                state["extra"] = source["h"] * alpha
                logits = receiver(record["ids"], use_cache=False).logits[:, -SCORED - 1 : -1].float()
                target = record["ids"][:, -SCORED:]
                losses = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
                )
                rows.append({
                    "sha256": record["sha256"],
                    "nll": losses.mean().item(),
                    "top1": (logits.argmax(-1) == target).float().mean().item(),
                })
    finally:
        handle.remove()
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "nll": sum(x["nll"] for x in rows) / len(rows),
        "top1": sum(x["top1"] for x in rows) / len(rows),
    }


def run_full_sender(sender, prepared: list[dict]):
    rows = []
    with torch.inference_mode():
        for record in prepared:
            ids = record["full_ids"]
            logits = sender(ids, use_cache=False).logits[:, -SCORED - 1 : -1].float()
            target = ids[:, -SCORED:]
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
            )
            rows.append({
                "sha256": record["sha256"],
                "nll": losses.mean().item(),
                "top1": (logits.argmax(-1) == target).float().mean().item(),
            })
    return rows


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
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
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
    train = prepare(sender, tokenizer, select(tokenizer, docs, "train", TRAIN_DOCS))
    dev = prepare(sender, tokenizer, select(tokenizer, docs, "dev", DEV_DOCS))

    train_grid = []
    for alpha in ALPHAS:
        rows = run_injected(receiver, train, alpha)
        train_grid.append({"alpha": alpha, **summarize(rows)})
    chosen = min(train_grid, key=lambda x: (x["nll"], abs(x["alpha"]), x["alpha"]))

    neutral_rows = run_injected(receiver, dev, 0.0)
    chosen_rows = run_injected(receiver, dev, chosen["alpha"])
    shuffled_rows = run_injected(receiver, dev, chosen["alpha"], shuffled=True)
    full_sender_rows = run_full_sender(sender, dev)
    gains = [a["nll"] - b["nll"] for a, b in zip(neutral_rows, chosen_rows)]
    content_advantage = [a["nll"] - b["nll"] for a, b in zip(shuffled_rows, chosen_rows)]
    gain_ci = bootstrap_ci(gains, SEED)
    content_ci = bootstrap_ci(content_advantage, SEED + 1)
    headroom = sum(
        neutral["nll"] - full["nll"] for neutral, full in zip(neutral_rows, full_sender_rows)
    ) / len(dev)
    fraction = (sum(gains) / len(gains)) / headroom if headroom > 0 else float("nan")
    finite = all(
        math.isfinite(x[key])
        for rows in (neutral_rows, chosen_rows, shuffled_rows, full_sender_rows)
        for x in rows for key in ("nll", "top1")
    )
    signal = finite and gain_ci[0] > 0 and content_ci[0] > 0
    sufficient = signal and fraction >= 0.20 and chosen["alpha"] not in (ALPHAS[0], ALPHAS[-1])
    result = {
        "seed": SEED,
        "corpus_sha256": sha(args.corpus.read_bytes()),
        "split_document_hashes_sha256": split_hashes,
        "test_tokenized_or_scored": False,
        "train_documents": TRAIN_DOCS,
        "dev_documents": DEV_DOCS,
        "prefix_tokens": PREFIX,
        "suffix_tokens": SUFFIX,
        "scored_suffix_tokens": SCORED,
        "bridge": "layer_norm_plus_scalar_additive",
        "trainable_parameter_count": 1,
        "alpha_grid_frozen": list(ALPHAS),
        "train_grid": train_grid,
        "chosen_alpha": chosen["alpha"],
        "dev": {
            "neutral": summarize(neutral_rows),
            "chosen_scalar": summarize(chosen_rows),
            "shuffled_document": summarize(shuffled_rows),
            "full_context_sender": summarize(full_sender_rows),
            "receiver_to_sender_headroom_nll": headroom,
            "fraction_of_headroom_closed": fraction,
            "neutral_minus_scalar_nll": sum(gains) / len(gains),
            "neutral_minus_scalar_bootstrap_95pct_ci": gain_ci,
            "shuffled_minus_scalar_nll": sum(content_advantage) / len(content_advantage),
            "shuffled_minus_scalar_bootstrap_95pct_ci": content_ci,
        },
        "finite": finite,
        "verdict": (
            "SCALAR_SUFFICIENT" if sufficient else
            "SCALAR_SIGNAL_BUT_INSUFFICIENT" if signal else
            "SCALAR_INSUFFICIENT"
        ),
        "per_document_dev": {
            "neutral": neutral_rows,
            "chosen_scalar": chosen_rows,
            "shuffled_document": shuffled_rows,
            "full_context_sender": full_sender_rows,
        },
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
