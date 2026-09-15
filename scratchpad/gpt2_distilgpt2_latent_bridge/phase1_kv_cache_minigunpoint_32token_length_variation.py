#!/usr/bin/env python3
"""Sequence-length robustness check for the frozen ordering adapter.

Cheap inference-only measurement, same harness discipline as the other
diagnostic scripts in this directory: no gradient step, frozen bridge. This
one does NOT reuse the original 32-dev-document split (documents at a
different length are necessarily different documents); it generates fresh
documents at each alternate length under the same generation rule and the
same underlying SEED, using the reusable `bucket`/`select_docs` machinery
from the ablations module, and checks the frozen adapter's classification
accuracy on them.

Motivation: paper.md Section 7 names "systematic length degradation" as an
absent robustness control, following the domain/scale-sensitivity spirit of
the closed-form KV-transfer related work (arXiv:2608.03893). The
`LowRankKVAdapter` correction is applied independently to every cache
position (Section 5.1's template-family check already establishes this
architecturally), so nothing requires the exact trained sequence length of
32 tokens; this makes it meaningful to ask whether the adapter's global
majority-rule classification still works at other lengths, held to the
same task definition (first-half-majority vs second-half-majority, always
an exact 50/50 total split).

Note: changing sequence length also changes where the receiver's own query
token ("Class:") lands in absolute position space (shorter sequences push it
EARLIER than the 35-token trained position; the existing persistence-delay
control already tests pushing it LATER). This confound is not separated
from "length" here; the check answers "does accuracy hold at this length,"
not "why."
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "phase1_kv_cache_minigunpoint_32token_ablations",
    HERE / "phase1_kv_cache_minigunpoint_32token_ablations.py",
)
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)  # only defines functions/classes; no main() runs

# (seq_len, k_range) pairs; k_range scaled proportionally from the original
# K_RANGE=(10,11,12,13,14) at HALF=16 (62.5%-87.5% majority margin).
LENGTH_CONFIGS = {
    16: (5, 6, 7),                    # HALF=8,  62.5%-87.5% -> 5-7 of 8
    48: (15, 16, 17, 18, 19, 20, 21),  # HALF=24, 62.5%-87.5% -> 15-21 of 24
}
DOCS_PER_LENGTH = 32


def make_sequence_len(rng: random.Random, cls: int, seq_len: int, k_range: tuple[int, ...]) -> list[str]:
    half = seq_len // 2
    k = rng.choice(k_range)
    if cls == 0:
        first_a, first_b = k, half - k
        second_a, second_b = half - k, k
    else:
        first_a, first_b = half - k, k
        second_a, second_b = k, half - k
    first = ["A"] * first_a + ["B"] * first_b
    second = ["A"] * second_a + ["B"] * second_b
    rng.shuffle(first)
    rng.shuffle(second)
    seq = first + second
    assert seq.count("A") == half and seq.count("B") == half
    return seq


def generate_pool_len(seq_len: int, k_range: tuple[int, ...], target_per_class: int) -> list[tuple[list[str], int]]:
    rng = random.Random(abl.SEED)
    pool: list[tuple[list[str], int]] = []
    seen: set[str] = set()
    for cls in (0, 1):
        made = 0
        attempts = 0
        while made < target_per_class and attempts < target_per_class * 50:
            attempts += 1
            seq = make_sequence_len(rng, cls, seq_len, k_range)
            key = "".join(seq)
            if key in seen:
                continue
            seen.add(key)
            pool.append((seq, cls))
            made += 1
        if made < target_per_class:
            raise RuntimeError(f"could not generate {target_per_class} unique class-{cls} sequences at length {seq_len}")
    return pool


def build_batch_len(sender, tokenizer, docs, sender_layer_selection):
    tokenized = [abl.seq_to_ids(tokenizer, seq) for seq, _ in docs]
    lens = {len(t) for t in tokenized}
    assert len(lens) == 1, f"prefix length not constant: {lens}"
    ids = torch.tensor(tokenized, dtype=torch.long, device=abl.DEVICE)
    with torch.no_grad():
        out = sender(ids, use_cache=True)
    pkv = out.past_key_values
    keys = [pkv.layers[i].keys.clone().detach() for i in sender_layer_selection]
    values = [pkv.layers[i].values.clone().detach() for i in sender_layer_selection]
    target_ids = [tokenizer.encode((" 0", " 1")[cls], add_special_tokens=False)[0] for _, cls in docs]
    target = torch.tensor(target_ids, dtype=torch.long, device=abl.DEVICE)
    return keys, values, target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--bridge", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(abl.SEED)
    tokenizer = abl.AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = abl.AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval().to(abl.DEVICE)
    receiver = abl.AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval().to(abl.DEVICE)
    for mdl in (sender, receiver):
        for p in mdl.parameters():
            p.requires_grad_(False)

    ckpt = torch.load(args.bridge, map_location=abl.DEVICE, weights_only=False)
    sender_layer_selection = ckpt["sender_layer_selection"]
    n_layers = len(sender_layer_selection)

    adapter = abl.LowRankKVAdapter(n_layers, ckpt["head_dim"], ckpt["n_head"], ckpt["rank"], seed=abl.SEED)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.to(abl.DEVICE)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)

    # Reproduce the trained-length (32) baseline for reference, exactly as in ablations.
    pool32 = abl.generate_pool(target_per_class=200)
    dev_docs32 = abl.select_docs(pool32, "dev", abl.DEV_DOCS)
    dev_keys_set = sorted({"".join(s) for s, _ in dev_docs32})
    assert dev_keys_set == ckpt["dev_docs_keys"], "trained-length dev split does not match the checkpointed training run"
    dev_k32, dev_v32, dev_target32 = abl.build_batch(sender, tokenizer, dev_docs32, sender_layer_selection)
    dq32 = abl.query_ids(tokenizer, "Class:", len(dev_docs32))
    with torch.no_grad():
        ak32 = adapter.forward_keys(dev_k32)
        av32 = adapter.forward_values(dev_v32)
        logits32 = abl.next_token_logits(receiver, dq32, abl.make_cache(ak32, av32))
        nll32, top1_32 = abl.nll_top1(logits32, dev_target32)

    results = {"trained_length_32": abl.summarize(nll32, top1_32)}

    for seq_len, k_range in LENGTH_CONFIGS.items():
        pool = generate_pool_len(seq_len, k_range, target_per_class=200)
        docs = abl.select_docs(pool, "dev", DOCS_PER_LENGTH)
        k, v, target = build_batch_len(sender, tokenizer, docs, sender_layer_selection)
        dq = abl.query_ids(tokenizer, "Class:", len(docs))
        with torch.no_grad():
            ak = adapter.forward_keys(k)
            av = adapter.forward_values(v)
            logits = abl.next_token_logits(receiver, dq, abl.make_cache(ak, av))
            nll, top1 = abl.nll_top1(logits, target)
        results[f"length_{seq_len}"] = abl.summarize(nll, top1)
        print(f"length={seq_len}: acc={results[f'length_{seq_len}']['accuracy']:.4f} nll={results[f'length_{seq_len}']['nll']:.4f}")

    result = {
        "mechanism": "sequence_length_robustness_check",
        "method": (
            "Fresh documents generated at each alternate sequence length under the "
            "same majority-rule task definition and generation SEED, scored through "
            "the frozen ordering adapter and receiver. Not a substitution on the "
            "original dev documents (different length implies different documents). "
            "K-range scaled proportionally to preserve the trained 62.5%-87.5% "
            "majority margin at each half-length."
        ),
        "lengths_tested": list(LENGTH_CONFIGS.keys()),
        "docs_per_length": DOCS_PER_LENGTH,
        **results,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
