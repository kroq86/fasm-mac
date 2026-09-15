#!/usr/bin/env python3
"""Alternate-symbol template-family robustness check for the frozen ordering
adapter.

Cheap inference-only measurement, same harness discipline as the other
_ablations/_moment_matched/_latency_baseline/_attention_cosine scripts
(loaded as a module, not duplicated): no gradient step, frozen bridge, dev
split verified against the checkpoint's own dev_docs_keys before scoring.

Motivation: paper.md Section 7 names "independent template families" as an
absent robustness control, following the domain-sensitivity methodology in
the closed-form KV-transfer related work (arXiv:2608.03893), which measures
how much a different surface domain costs a fixed transfer mechanism. The
adapter here is a per-token low-rank correction (LowRankKVAdapter applies
the same linear map independently to every cache position, regardless of
which token produced it), so nothing in its architecture requires the exact
training vocabulary -- this makes it mechanically meaningful to ask whether
it still works if the SAME 32 dev documents are rendered with a different,
single-BPE-token symbol pair (X/Y) instead of the trained pair (A/B),
holding the underlying majority-rule structure, class labels, and specific
per-document arrangement (which position gets which symbol) fixed. This
isolates surface vocabulary as the only changed variable; it is not a fresh
random draw of new documents.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "phase1_kv_cache_minigunpoint_32token_ablations",
    HERE / "phase1_kv_cache_minigunpoint_32token_ablations.py",
)
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)  # only defines functions/classes; no main() runs

ALT_SYMBOL_MAP = {"A": "X", "B": "Y"}


def seq_to_ids_relabeled(tokenizer, seq, symbol_map):
    text = "Sequence:" + "".join(f" {symbol_map.get(tok, tok)}" for tok in seq)
    return tokenizer.encode(text, add_special_tokens=False)


def build_batch_relabeled(sender, tokenizer, docs, sender_layer_selection, symbol_map):
    tokenized = [seq_to_ids_relabeled(tokenizer, seq, symbol_map) for seq, _ in docs]
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
    for m in (sender, receiver):
        for p in m.parameters():
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

    pool = abl.generate_pool(target_per_class=200)
    dev_docs = abl.select_docs(pool, "dev", abl.DEV_DOCS)
    dev_keys_set = sorted({"".join(s) for s, _ in dev_docs})
    assert dev_keys_set == ckpt["dev_docs_keys"], "dev split does not match the checkpointed training run"

    # Verify the alternate symbols each tokenize to exactly one id, same as A/B,
    # so the relabeled sequence has the same token count as the original.
    for sym in ALT_SYMBOL_MAP.values():
        ids = tokenizer.encode(f" {sym}", add_special_tokens=False)
        assert len(ids) == 1, f"alternate symbol {sym!r} is not a single BPE token: {ids}"

    dev_k, dev_v, dev_target = abl.build_batch(sender, tokenizer, dev_docs, sender_layer_selection)
    dq = abl.query_ids(tokenizer, "Class:", len(dev_docs))

    alt_k, alt_v, alt_target = build_batch_relabeled(sender, tokenizer, dev_docs, sender_layer_selection, ALT_SYMBOL_MAP)
    assert torch.equal(dev_target, alt_target), "relabeling must not change the target class sequence"

    with torch.no_grad():
        ak = adapter.forward_keys(dev_k)
        av = adapter.forward_values(dev_v)
        correct_logits = abl.next_token_logits(receiver, dq, abl.make_cache(ak, av))
        correct_nll, correct_top1 = abl.nll_top1(correct_logits, dev_target)

        alt_ak = adapter.forward_keys(alt_k)
        alt_av = adapter.forward_values(alt_v)
        alt_logits = abl.next_token_logits(receiver, dq, abl.make_cache(alt_ak, alt_av))
        alt_nll, alt_top1 = abl.nll_top1(alt_logits, alt_target)

    result = {
        "mechanism": "template_family_robustness_check",
        "method": (
            "Same 32 original dev documents, same class labels and per-document "
            "A/B arrangement, rendered with an alternate single-BPE-token symbol "
            "pair (A->X, B->Y) instead of the trained pair (A/B). The frozen "
            "sender, adapter, and receiver are otherwise unchanged; no retraining."
        ),
        "trained_symbols": {"class0_majority": "A", "class1_majority": "B"},
        "alternate_symbols": ALT_SYMBOL_MAP,
        "trained_template": abl.summarize(correct_nll, correct_top1),
        "alternate_template": abl.summarize(alt_nll, alt_top1),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
