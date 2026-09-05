#!/usr/bin/env python3
"""KV-cache handoff, not embedding-space latent prefix.

Every prior positive/negative result in this investigation used
hidden-states -> linear translator -> continuous embedding-space prefix.
DistilGPT2 had to absorb those foreign vectors into its own internal
representation once, at the start, via ordinary attention over a fixed
prefix. This script tests a mechanically different transfer: GPT-2's own
computed KV-cache (attention memory) for the prefix is transplanted
directly into DistilGPT2's attention mechanism via `past_key_values`, so
DistilGPT2 can query into it fresh at every generated position, the way a
model queries its own cache. Shapes are compatible without any adapter:
both models use hidden width 768, 12 attention heads, head_dim 64.

DistilGPT2 is not an unrelated model -- it was distilled from this exact
GPT-2 124M following the standard Distil* recipe (Sanh et al. 2019),
whose student is initialized from a stride-2 subset of the teacher's
layers (0-indexed 1,3,5,7,9,11 of GPT-2's 12). That is the layer selection
used here for the raw (untrained) passthrough test: if any pair of
frozen, independently-named models could have naturally-aligned Q/K
subspaces without training an adapter, it is this one, precisely because
of how the student's weights originated.

Claim: raw, training-free KV-cache transplant (no adapter at all) shows a
correct-vs-shuffled gate pass on the same controlled single-token fact
task that already succeeded via embedding-prefix (81.25% top-1). Falsified
by: correct indistinguishable from shuffled even with this mechanism --
which would mean the alignment DistilGPT2 has to GPT-2's KV space through
distillation is not sufficient on its own for cross-cache attention, and
a trained per-layer K/V adapter (a natural next rung, not run here) would
be needed.

Same task as phase1_controlled_fact_transfer.py: sender sees "The secret
code is {code}." (3-digit, single BPE token, verified); receiver sees only
"The secret code is" and must predict the next token, using nothing but
the transplanted KV-cache. Four arms: no_prefix (empty cache), zero_sender
(all-zero K/V at every selected layer -- architecturally forced null, not
trained-to-zero), correct, shuffled (wrong document's cache). Same
sha256-bucket disjoint train/dev code split. No training in this first
pass -- purely a mechanism/alignment test, not a bridge-family rung.
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
from transformers.cache_utils import DynamicCache

SEED = 20260904
DEV_CODES = 32
CODE_MIN, CODE_MAX = 100, 999
SENDER_LAYER_SELECTION = (1, 3, 5, 7, 9, 11)  # matches Distil* stride-2 init


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


def sender_cache(sender, tokenizer, code: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    ids = tokenizer.encode(f"The secret code is {code}.", return_tensors="pt")
    with torch.no_grad():
        out = sender(ids, use_cache=True)
    pkv = out.past_key_values
    keys = [pkv.layers[i].keys.clone() for i in SENDER_LAYER_SELECTION]
    values = [pkv.layers[i].values.clone() for i in SENDER_LAYER_SELECTION]
    return keys, values


def make_cache(keys: list[torch.Tensor] | None, values: list[torch.Tensor] | None, like: torch.Tensor | None = None) -> DynamicCache:
    cache = DynamicCache()
    if keys is None:
        return cache
    for k, v in zip(keys, values):
        cache.update(k.clone(), v.clone(), layer_idx=len(cache.layers))
    return cache


def zero_cache(shape_like_keys: list[torch.Tensor]) -> DynamicCache:
    cache = DynamicCache()
    for k in shape_like_keys:
        z = torch.zeros_like(k)
        cache.update(z, z.clone(), layer_idx=len(cache.layers))
    return cache


def nll_top1(receiver, suffix_ids: torch.Tensor, cache: DynamicCache, target_id: int) -> tuple[float, float]:
    with torch.no_grad():
        out = receiver(suffix_ids, past_key_values=cache, use_cache=False)
    logits = out.logits[:, -1, :]
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

    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval()
    for m in (sender, receiver):
        for p in m.parameters():
            p.requires_grad_(False)

    dev_codes = select_codes(tokenizer, "dev", DEV_CODES)
    suffix_ids = tokenizer.encode("The secret code is", return_tensors="pt")

    dev = []
    for code in dev_codes:
        keys, values = sender_cache(sender, tokenizer, code)
        target_id = tokenizer.encode(f" {code}", add_special_tokens=False)
        assert len(target_id) == 1
        dev.append({"code": code, "keys": keys, "values": values, "target_id": target_id[0]})

    n = len(dev)
    no_prefix_rows, zero_sender_rows, chosen_rows, shuffled_rows = [], [], [], []
    for i, d in enumerate(dev):
        nll, top1 = nll_top1(receiver, suffix_ids, make_cache(None, None), d["target_id"])
        no_prefix_rows.append({"code": d["code"], "nll": nll, "top1": top1})

        nll, top1 = nll_top1(receiver, suffix_ids, zero_cache(d["keys"]), d["target_id"])
        zero_sender_rows.append({"code": d["code"], "nll": nll, "top1": top1})

        nll, top1 = nll_top1(receiver, suffix_ids, make_cache(d["keys"], d["values"]), d["target_id"])
        chosen_rows.append({"code": d["code"], "nll": nll, "top1": top1})

        other = dev[(i - 1) % n]
        nll, top1 = nll_top1(receiver, suffix_ids, make_cache(other["keys"], other["values"]), d["target_id"])
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
        "seed": SEED, "sender_layer_selection": list(SENDER_LAYER_SELECTION),
        "dev_codes": dev_codes, "trainable_parameter_count": 0,
        "mechanism": "raw_untrained_kv_cache_transplant",
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
        "verdict": "KV_CACHE_RAW_TRANSFER_SUPPORTED" if signal else "KV_CACHE_RAW_NO_TRANSFER",
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
