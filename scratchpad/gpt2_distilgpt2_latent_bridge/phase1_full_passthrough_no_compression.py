#!/usr/bin/env python3
"""Separate preregistered instrument: is lossy compression (64 sender
positions -> 8 pooled latent tokens) the reason latent-prefix fusion
failed, rather than the fusion mechanism itself?

Claim: transferring ALL 64 withheld-prefix positions' final-layer sender
states directly (norm-calibrated, no learned pooling/compression at all --
zero trainable parameters) shows a correct-vs-shuffled document-specific
signal where the 8-token lossy-pooled version did not. This is the
richest-bandwidth version of prefix-token fusion possible without
changing site or mechanism: maximum available sequence information, no
compression bottleneck to blame. Falsified by: correct-vs-shuffled CI
still including zero even at full bandwidth -- which would mean
compression wasn't the bottleneck either, making the remaining candidates
(multi-layer representations, state-delta trajectories, full KV-cache)
progressively less likely to help and progressively more expensive to
test, per the project's stop rule (cheapest, most direct test first).

No training: each of the 64 sender prefix positions' LayerNorm'd
final-layer hidden state is individually rescaled to the receiver's real
median token-embedding norm (same target_norm computation as the
norm-calibrated projector rung), then all 64 are prepended directly to
DistilGPT2's input embedding sequence -- pure evaluation, no optimizer,
no epochs. Same corpus/splits/site/scoring window/receiver-matched
ceiling as the rest of Phase 1. Four arms: no_prefix (K=0), zero_sender
(64 zero vectors), correct (64 real calibrated states), shuffled (64 real
calibrated states from a different, i-1-rotated document). Test remains
closed.
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
SENDER_LAYER = 12


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


def prepare(sender, tokenizer, records: list[dict], target_norm: float) -> list[dict]:
    prepared = []
    with torch.no_grad():
        for record in records:
            tokens = record["ids"]
            prefix_ids = torch.tensor([tokens[:PREFIX]], dtype=torch.long)
            suffix_ids = torch.tensor([tokens[PREFIX:]], dtype=torch.long)
            out = sender(prefix_ids, use_cache=False, output_hidden_states=True)
            H = out.hidden_states[SENDER_LAYER].float()  # (1, PREFIX, 768)
            H = torch.nn.functional.layer_norm(H, (H.shape[-1],))
            norm = H.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            H_calibrated = (H / norm * target_norm).squeeze(0)  # (PREFIX, 768)
            prepared.append({"sha256": record["sha256"], "suffix_ids": suffix_ids, "H": H_calibrated})
    return prepared


def receiver_logits(receiver, wte, suffix_ids: torch.Tensor, latent: torch.Tensor | None):
    suffix_embeds = wte(suffix_ids)
    if latent is None:
        combined = suffix_embeds
    else:
        combined = torch.cat([latent.unsqueeze(0), suffix_embeds], dim=1)
    return receiver(inputs_embeds=combined, use_cache=False).logits


def per_doc_nll_top1(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred = logits[:, -SCORED - 1 : -1].float()
    losses = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none"
    )
    top1 = (pred.argmax(-1) == target).float().mean().item()
    return losses.mean().item(), top1


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
    ap.add_argument("--ceiling", type=Path, required=True)
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

    docs = load_documents(args.corpus)
    corpus_sha256 = sha(args.corpus.read_bytes())
    dev_records = select(tokenizer, docs, "dev", DEV_DOCS)
    dev = prepare(sender, tokenizer, dev_records, target_norm)

    ceiling = json.loads(args.ceiling.read_text())
    full_rows = ceiling["per_document"]["receiver_full_context"]

    n = len(dev)
    with torch.no_grad():
        no_prefix_rows, zero_sender_rows, chosen_rows, shuffled_rows = [], [], [], []
        for i, d in enumerate(dev):
            target = d["suffix_ids"][:, -SCORED:]

            logits = receiver_logits(receiver, wte, d["suffix_ids"], None)
            nll, top1 = per_doc_nll_top1(logits, target)
            no_prefix_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

            zero_latent = torch.zeros(PREFIX, 768)
            logits = receiver_logits(receiver, wte, d["suffix_ids"], zero_latent)
            nll, top1 = per_doc_nll_top1(logits, target)
            zero_sender_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

            logits = receiver_logits(receiver, wte, d["suffix_ids"], d["H"])
            nll, top1 = per_doc_nll_top1(logits, target)
            chosen_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

            other_H = dev[(i - 1) % n]["H"]
            logits = receiver_logits(receiver, wte, d["suffix_ids"], other_H)
            nll, top1 = per_doc_nll_top1(logits, target)
            shuffled_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

    assert [r["sha256"] for r in no_prefix_rows] == [r["sha256"] for r in full_rows], \
        "dev document set/order must match the receiver-ceiling artifact exactly"

    gains = [a["nll"] - c["nll"] for a, c in zip(no_prefix_rows, chosen_rows)]
    content_advantage = [s["nll"] - c["nll"] for s, c in zip(shuffled_rows, chosen_rows)]
    zero_advantage = [z["nll"] - c["nll"] for z, c in zip(zero_sender_rows, chosen_rows)]
    headroom = [a["nll"] - f["nll"] for a, f in zip(no_prefix_rows, full_rows)]
    gain_mean = sum(gains) / len(gains)
    headroom_mean = sum(headroom) / len(headroom)
    fraction = gain_mean / headroom_mean if headroom_mean > 0 else float("nan")

    finite = all(
        math.isfinite(r["nll"]) and math.isfinite(r["top1"])
        for rows in (no_prefix_rows, zero_sender_rows, chosen_rows, shuffled_rows) for r in rows
    )
    gain_ci = bootstrap_ci(gains, SEED)
    content_ci = bootstrap_ci(content_advantage, SEED + 1)
    zero_ci = bootstrap_ci(zero_advantage, SEED + 2)
    signal = finite and content_ci[0] > 0

    result = {
        "seed": SEED, "corpus_sha256": corpus_sha256, "test_tokenized_or_scored": False,
        "dev_documents": DEV_DOCS, "sender_layer": SENDER_LAYER,
        "num_latent_tokens": PREFIX,
        "bridge": "full_passthrough_no_compression_norm_calibrated_no_training",
        "target_norm_from_receiver_embedding_median": target_norm,
        "trainable_parameter_count": 0,
        "dev": {
            "no_prefix": {"nll": sum(r["nll"] for r in no_prefix_rows) / n, "top1": sum(r["top1"] for r in no_prefix_rows) / n},
            "zero_sender": {"nll": sum(r["nll"] for r in zero_sender_rows) / n, "top1": sum(r["top1"] for r in zero_sender_rows) / n},
            "correct": {"nll": sum(r["nll"] for r in chosen_rows) / n, "top1": sum(r["top1"] for r in chosen_rows) / n},
            "shuffled_document": {"nll": sum(r["nll"] for r in shuffled_rows) / n, "top1": sum(r["top1"] for r in shuffled_rows) / n},
            "receiver_matched_headroom_nll": headroom_mean,
            "fraction_of_receiver_matched_headroom_closed": fraction,
            "no_prefix_minus_correct_nll": gain_mean,
            "no_prefix_minus_correct_bootstrap_95pct_ci": gain_ci,
            "shuffled_minus_correct_nll": sum(content_advantage) / len(content_advantage),
            "shuffled_minus_correct_bootstrap_95pct_ci": content_ci,
            "zero_sender_minus_correct_nll": sum(zero_advantage) / len(zero_advantage),
            "zero_sender_minus_correct_bootstrap_95pct_ci": zero_ci,
        },
        "finite": finite,
        "verdict": (
            "FULL_PASSTHROUGH_DOCUMENT_SPECIFIC_TRANSFER" if signal else
            "FULL_PASSTHROUGH_NO_DOCUMENT_SPECIFIC_TRANSFER"
        ),
        "per_document_dev": {
            "no_prefix": no_prefix_rows, "zero_sender": zero_sender_rows,
            "correct": chosen_rows, "shuffled_document": shuffled_rows,
        },
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(f"gain={gain_mean:.6f} gain_ci={gain_ci} shuffled_minus_correct={result['dev']['shuffled_minus_correct_nll']:.6f} "
          f"content_ci={content_ci} zero_ci={zero_ci} fraction={fraction:.6f} verdict={result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
