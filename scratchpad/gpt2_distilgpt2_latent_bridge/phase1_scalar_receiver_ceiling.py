#!/usr/bin/env python3
"""Metric correction, not a new experiment: recompute
fraction_of_headroom_closed against a receiver-matched ceiling.

The original phase1_scalar_sweep.py used GPT-2's (sender's) full-context
NLL as the ceiling for "headroom". That conflates two different gaps:
information the receiver is missing, and DistilGPT2 being a strictly
smaller/weaker model than GPT-2 regardless of information. This script
adds the one missing arm -- DistilGPT2 itself on the real, full
prefix+suffix tokens -- on the IDENTICAL 32 dev documents (verified by
sha256 against the existing artifact) and recomputes the fraction against
that receiver-matched ceiling instead.

No training. No new documents. No test-split access. Reuses the exact
document-selection logic (bucket/select) from phase1_scalar_sweep.py so
the reconstructed dev set is provably the same 32 documents.
"""
from __future__ import annotations

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
CORPUS = Path("/tmp/fasm-latent-bridge-data/TinyStories-valid.txt")
RECEIVER = "/Users/ll/distilgpt2"
EXISTING = Path("/Users/ll/fasm-mac/scratchpad/gpt2_distilgpt2_latent_bridge/phase1_scalar_result.json")
OUT = Path("/Users/ll/fasm-mac/scratchpad/gpt2_distilgpt2_latent_bridge/phase1_scalar_receiver_ceiling.json")


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


def select_dev(tokenizer, docs: list[str], count: int) -> list[dict]:
    chosen = []
    for doc in sorted((x for x in docs if bucket(x) == "dev"), key=lambda x: sha(x.encode())):
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if len(ids) >= PREFIX + SUFFIX:
            chosen.append({"sha256": sha(doc.encode()), "ids": ids[: PREFIX + SUFFIX]})
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"only {len(chosen)} eligible dev documents")
    return chosen


def bootstrap_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return [means[249], means[9749]]


def main() -> int:
    existing = json.loads(EXISTING.read_text())
    assert sha(CORPUS.read_bytes()) == existing["corpus_sha256"], "corpus mismatch"

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(RECEIVER, local_files_only=True)
    receiver = AutoModelForCausalLM.from_pretrained(RECEIVER, local_files_only=True).eval()
    for p in receiver.parameters():
        p.requires_grad_(False)

    docs = load_documents(CORPUS)
    dev = select_dev(tokenizer, docs, DEV_DOCS)
    # split_document_hashes_sha256["dev"] in the original artifact hashes
    # the entire dev BUCKET (thousands of eligible documents), not just the
    # 32 selected ones -- comparing against it directly is a category
    # mismatch, not a real discrepancy (confirmed by a direct set
    # comparison during debugging). The correct verification is that the
    # 32 selected documents' sha256 set matches per_document_dev exactly.
    existing_order = [r["sha256"] for r in existing["per_document_dev"]["neutral"]]
    computed_set = {r["sha256"] for r in dev}
    assert computed_set == set(existing_order), "reconstructed 32 dev documents do not match existing artifact"

    # reorder to match existing_order exactly for a correct paired comparison
    by_sha = {r["sha256"]: r for r in dev}
    dev = [by_sha[s] for s in existing_order]
    assert [r["sha256"] for r in dev] == existing_order

    receiver_full_rows = []
    with torch.inference_mode():
        for record in dev:
            full_ids = torch.tensor([record["ids"]], dtype=torch.long)  # prefix+suffix, real tokens, receiver's own forward
            logits = receiver(full_ids, use_cache=False).logits[:, -SCORED - 1 : -1].float()
            target = full_ids[:, -SCORED:]
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
            )
            receiver_full_rows.append({
                "sha256": record["sha256"],
                "nll": losses.mean().item(),
                "top1": (logits.argmax(-1) == target).float().mean().item(),
            })

    neutral_rows = existing["per_document_dev"]["neutral"]
    chosen_rows = existing["per_document_dev"]["chosen_scalar"]
    assert [r["sha256"] for r in neutral_rows] == [r["sha256"] for r in receiver_full_rows]
    assert [r["sha256"] for r in chosen_rows] == [r["sha256"] for r in receiver_full_rows]

    finite = all(math.isfinite(r["nll"]) and math.isfinite(r["top1"]) for r in receiver_full_rows)

    old_headroom = [n["nll"] - f["nll"] for n, f in zip(neutral_rows, existing["per_document_dev"]["full_context_sender"])]
    new_headroom = [n["nll"] - f["nll"] for n, f in zip(neutral_rows, receiver_full_rows)]
    gains = [n["nll"] - c["nll"] for n, c in zip(neutral_rows, chosen_rows)]

    old_headroom_mean = sum(old_headroom) / len(old_headroom)
    new_headroom_mean = sum(new_headroom) / len(new_headroom)
    gain_mean = sum(gains) / len(gains)

    old_fraction = existing["dev"]["fraction_of_headroom_closed"]
    new_fraction = gain_mean / new_headroom_mean if new_headroom_mean > 0 else float("nan")

    # per-document fraction distribution + bootstrap CI on the NEW fraction,
    # via the ratio of paired bootstrap means (gain_mean / headroom_mean per
    # resample) -- more honest than a single point estimate alone.
    rng = random.Random(SEED)
    n = len(dev)
    fraction_resamples = []
    for _ in range(10000):
        idx = [rng.randrange(n) for _ in range(n)]
        g = sum(gains[i] for i in idx) / n
        h = sum(new_headroom[i] for i in idx) / n
        if h > 0:
            fraction_resamples.append(g / h)
    fraction_resamples.sort()
    fraction_ci = [fraction_resamples[249], fraction_resamples[9749]] if len(fraction_resamples) >= 9750 else None

    negative_new_headroom_docs = sum(1 for h in new_headroom if h <= 0)

    result = {
        "purpose": "metric correction only -- same dev documents, same scalar bridge output, new ceiling",
        "seed": SEED,
        "corpus_sha256": existing["corpus_sha256"],
        "dev_document_set_verified": computed_set == set(existing_order),
        "old_ceiling": "GPT-2 (sender) full-context NLL -- conflates information gap with sender/receiver capacity gap",
        "new_ceiling": "DistilGPT2 (receiver) full-context NLL -- same model as neutral/chosen arms, isolates information gap",
        "old_headroom_mean_nll": old_headroom_mean,
        "new_headroom_mean_nll": new_headroom_mean,
        "old_fraction_of_headroom_closed": old_fraction,
        "new_fraction_of_headroom_closed": new_fraction,
        "new_fraction_bootstrap_95pct_ci": fraction_ci,
        "gain_mean_nll_unchanged_from_original": gain_mean,
        "receiver_full_context": {
            "nll": sum(r["nll"] for r in receiver_full_rows) / len(receiver_full_rows),
            "top1": sum(r["top1"] for r in receiver_full_rows) / len(receiver_full_rows),
        },
        "documents_where_new_headroom_nonpositive": negative_new_headroom_docs,
        "finite": finite,
        "per_document": {
            "receiver_full_context": receiver_full_rows,
            "old_headroom_per_doc": old_headroom,
            "new_headroom_per_doc": new_headroom,
            "gain_per_doc": gains,
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    OUT.write_text(encoded)
    print(f"old_fraction={old_fraction:.6f} new_fraction={new_fraction:.6f} "
          f"old_headroom={old_headroom_mean:.6f} new_headroom={new_headroom_mean:.6f} "
          f"gain={gain_mean:.6f} negative_new_headroom_docs={negative_new_headroom_docs}/{n}")
    if fraction_ci:
        print(f"new_fraction_95pct_ci=[{fraction_ci[0]:.6f}, {fraction_ci[1]:.6f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
