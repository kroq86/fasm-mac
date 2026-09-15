#!/usr/bin/env python3
"""Attention-output cosine diagnostic for the K-vs-V asymmetry.

Cheap inference-only measurement, same harness discipline as the other
_ablations/_moment_matched/_latency_baseline scripts (loaded as a module,
not duplicated): no gradient step, frozen bridge, dev split verified against
the checkpoint's own dev_docs_keys before scoring.

Motivation: the checkpointed ablations find "correct_K_wrong_V" (accuracy
0.125, 4/32) much worse than "wrong_K_correct_V" (accuracy 0.875, 28/32),
and paper.md states this "does not directly measure query-key geometric
alignment." Cheng et al.'s companion paper on cross-model KV transfer
(arXiv:2608.03893) reports that raw regression fit poorly predicts transfer
success (r=-0.20), while attention-output cosine similarity between the
true and perturbed attention output correlates well with retention
(r=+0.57): a mechanistic, not merely behavioral, explanation. This script
applies that same instrument here: for each receiver layer, it extracts the
real attention weights (via output_attentions=True, eager attention) at the
final query position -- the one position whose hidden state determines the
reported class-token logits -- and recomputes the cache's contribution to
the attention output as (attention weights over cache positions) @ (V used
in that condition). It then measures cosine similarity of that vector,
per layer, between the correct-cache condition and each of the two K/V-swap
conditions, to test whether the condition with lower cosine similarity to
the correct attention output is the one with worse task accuracy.

This is a 2-condition qualitative diagnostic on one checkpoint, not a
correlation study over many examples/checkpoints; report language reflects
that.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "phase1_kv_cache_minigunpoint_32token_ablations",
    HERE / "phase1_kv_cache_minigunpoint_32token_ablations.py",
)
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)  # only defines functions/classes; no main() runs
abl.DEVICE = "cpu"  # eager attention + output_attentions is simplest to reason about on CPU


def cache_attention_output(receiver, q_ids, keys, values, cache_len):
    """Run one forward pass with output_attentions=True and return, per
    layer, the cache's contribution to the attention output at the final
    query position: (attn weights over cache positions) @ (V at that layer).
    Shape per layer: [batch, heads, head_dim]."""
    cache = abl.make_cache(keys, values)
    out = receiver(q_ids, past_key_values=cache, use_cache=False, output_attentions=True)
    per_layer = []
    for layer_idx, attn in enumerate(out.attentions):
        w_last = attn[:, :, -1, :cache_len]  # [batch, heads, cache_len]
        v = values[layer_idx]  # [batch, heads, cache_len, head_dim]
        attn_out = torch.einsum("bhc,bhcd->bhd", w_last, v)
        per_layer.append(attn_out)
    return per_layer, out.logits[:, -1, :]


def cosine_per_layer(a_layers, b_layers):
    """Mean cosine similarity per layer, averaged over batch and heads."""
    results = []
    for a, b in zip(a_layers, b_layers):
        cos = F.cosine_similarity(a, b, dim=-1)  # [batch, heads]
        results.append(cos.mean().item())
    return results


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
    # eager attention implementation is required for output_attentions to be populated
    receiver = abl.AutoModelForCausalLM.from_pretrained(
        args.receiver, local_files_only=True, attn_implementation="eager"
    ).eval().to(abl.DEVICE)
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

    dev_k, dev_v, dev_target = abl.build_batch(sender, tokenizer, dev_docs, sender_layer_selection)
    dq = abl.query_ids(tokenizer, "Class:", len(dev_docs))
    cache_len = dev_k[0].shape[2]

    with torch.no_grad():
        ak = adapter.forward_keys(dev_k)
        av = adapter.forward_values(dev_v)
        sk, sv = abl.roll_batch(dev_k), abl.roll_batch(dev_v)
        ak_s = adapter.forward_keys(sk)
        av_s = adapter.forward_values(sv)

        correct_attn, correct_logits = cache_attention_output(receiver, dq, ak, av, cache_len)
        wrongK_attn, wrongK_logits = cache_attention_output(receiver, dq, ak_s, av, cache_len)  # wrong K, correct V
        wrongV_attn, wrongV_logits = cache_attention_output(receiver, dq, ak, av_s, cache_len)  # correct K, wrong V

        correct_nll, correct_top1 = abl.nll_top1(correct_logits, dev_target)
        wrongK_nll, wrongK_top1 = abl.nll_top1(wrongK_logits, dev_target)
        wrongV_nll, wrongV_top1 = abl.nll_top1(wrongV_logits, dev_target)

    cos_wrongK = cosine_per_layer(wrongK_attn, correct_attn)
    cos_wrongV = cosine_per_layer(wrongV_attn, correct_attn)

    result = {
        "mechanism": "attention_output_cosine_diagnostic",
        "method": (
            "For each of the 6 receiver layers, attention weights over the "
            "imported-cache positions at the final query token (output_attentions=True, "
            "eager attention) are contracted with the V used in that condition to give "
            "the cache's contribution to the attention output. Cosine similarity is "
            "measured per layer between the correct-cache condition and each K/V-swap "
            "condition, averaged over batch and heads. Cheng et al. (arXiv:2608.03893) "
            "report this measure correlates with cross-model transfer retention where raw "
            "regression fit does not; this is a 2-condition qualitative check on one "
            "checkpoint, not a many-point correlation study."
        ),
        "accuracy_reference": {
            "correct": abl.summarize(correct_nll, correct_top1),
            "wrong_K_correct_V": abl.summarize(wrongK_nll, wrongK_top1),
            "correct_K_wrong_V": abl.summarize(wrongV_nll, wrongV_top1),
            "source": "matches phase1_kv_cache_minigunpoint_32token_ablations_result.json k_vs_v arm",
        },
        "attention_output_cosine_to_correct": {
            "wrong_K_correct_V_per_layer": cos_wrongK,
            "correct_K_wrong_V_per_layer": cos_wrongV,
            "wrong_K_correct_V_mean": sum(cos_wrongK) / len(cos_wrongK),
            "correct_K_wrong_V_mean": sum(cos_wrongV) / len(cos_wrongV),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
