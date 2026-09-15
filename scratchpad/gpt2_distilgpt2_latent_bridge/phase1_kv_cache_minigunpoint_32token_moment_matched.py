#!/usr/bin/env python3
"""Moment-matched random-cache control for the frozen ordering adapter.

Cheap inference-only screening pass, same harness discipline as
phase1_kv_cache_minigunpoint_32token_ablations.py (loaded here as a module,
not duplicated): no gradient step, frozen bridge, dev split verified against
the checkpoint's own dev_docs_keys before scoring.

Motivation: the checkpointed ablations distinguish correct cache (0.96875
acc) from a different-document/opposite-class cache (0.03125 acc, i.e. the
"shuffled_document" arm) and from an all-zero cache ("zero_sender" arm, also
degenerate). Neither control isolates whether the effect is specific to
CONTENT identity as opposed to merely a shape/distribution artifact: a
different-document cache is itself a real, in-distribution cache (so a large
effect there could still just mean "any real cache other than the paired
one hurts"), and an all-zero cache is maximally out-of-distribution (so its
effect could be pure shape shock rather than absence of content). Cheng et
al. ("When Does Latent Communication Pay?", arXiv:2608.04893) name exactly
this gap and construct a moment-matched random control: Gaussian cache
entries centered and rescaled per example, per layer, and per K/V tensor to
match the true cache's own moments. This preserves the cache's per-tensor
statistics (so it is not a shape/scale shock like the zero arm) while
destroying every trace of sender content (so any residual accuracy above
chance cannot be attributed to leftover real information). paper.md
Section 7 already lists "moment-matched random-cache controls" as an
absent control and Section 2 already cites Cheng et al. as the precedent;
this script closes exactly that gap for the ordering task, without
retraining the adapter or touching the historical checkpointed result.
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


def moment_matched_like(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Gaussian noise per example (dim 0), matched to that example's own
    mean/std over all remaining dims of this one (layer, K-or-V) tensor."""
    b = tensor.shape[0]
    flat = tensor.reshape(b, -1)
    mean = flat.mean(dim=1).view(b, *([1] * (tensor.dim() - 1)))
    std = flat.std(dim=1).view(b, *([1] * (tensor.dim() - 1)))
    noise = torch.randn(tensor.shape, generator=generator, device=tensor.device)
    return noise * std + mean


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

    dev_k, dev_v, dev_target = abl.build_batch(sender, tokenizer, dev_docs, sender_layer_selection)
    dq = abl.query_ids(tokenizer, "Class:", len(dev_docs))

    ak = adapter.forward_keys(dev_k)
    av = adapter.forward_values(dev_v)
    correct_logits = abl.next_token_logits(receiver, dq, abl.make_cache(ak, av))
    correct_nll, correct_top1 = abl.nll_top1(correct_logits, dev_target)

    g = torch.Generator(device=abl.DEVICE).manual_seed(abl.SEED + 100)
    mm_k = [moment_matched_like(k, g) for k in ak]
    mm_v = [moment_matched_like(v, g) for v in av]
    mm_logits = abl.next_token_logits(receiver, dq, abl.make_cache(mm_k, mm_v))
    mm_nll, mm_top1 = abl.nll_top1(mm_logits, dev_target)

    result = {
        "mechanism": "moment_matched_random_cache_control",
        "method": (
            "Gaussian noise per example, per layer, per K/V tensor, matched "
            "to that tensor's own per-example mean/std (Cheng et al. "
            "arXiv:2608.04893 recipe), applied post-adapter on the same 32 "
            "original dev sequences as the checkpointed ablations run."
        ),
        "seed": abl.SEED,
        "correct": abl.summarize(correct_nll, correct_top1),
        "moment_matched_random": abl.summarize(mm_nll, mm_top1),
        "moment_matched_minus_correct": abl.delta_ci(correct_nll, mm_nll, abl.SEED + 101),
        "reference_historical_arms": {
            "zero_sender_accuracy": 0.0,
            "shuffled_document_accuracy": 0.03125,
            "source": "phase1_kv_cache_minigunpoint_32token_checkpoint_result.json / _ablations_result.json (not recomputed here)",
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
