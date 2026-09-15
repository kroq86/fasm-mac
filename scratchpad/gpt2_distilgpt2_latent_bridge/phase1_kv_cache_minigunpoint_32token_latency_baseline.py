#!/usr/bin/env python3
"""Re-prefill efficiency baseline for the frozen ordering adapter.

Cheap inference-only measurement, same harness discipline as the other
_ablations/_moment_matched scripts (loaded as a module, not duplicated): no
gradient step, frozen bridge, dev split verified against the checkpoint's own
dev_docs_keys before timing.

Motivation: paper.md Section 4 and Section 7 both name "raw-context
transmission and receiver re-prefill" as an essential efficiency baseline
that "has not been measured here." This script measures one instance of it
in the PyTorch reference environment (CPU, single-threaded, wall-clock),
NOT in the native FASM/Accelerate deployment path from Section 6 -- it
answers "is the cache-transfer path faster than just handing the receiver
the raw text and letting it prefill from scratch," not "how fast is the
native binary." Two paths are timed on the same 32 original dev sequences:

  (a) cache-transfer: sender prefills "Sequence: <32 symbols>" (12 layers),
      the frozen adapter projects the selected-layer K/V, the receiver
      consumes the adapted cache and processes only the query "Class:".
  (b) re-prefill: the receiver alone prefills "Sequence: <32 symbols> Class:"
      from scratch, with no sender, no adapter, and no imported cache.

Both paths end at the same point (receiver's next-token logits over the
class token); only the path to get there differs. CPU-only, single thread,
to avoid MPS dispatch-overhead noise dominating small-model timings.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "phase1_kv_cache_minigunpoint_32token_ablations",
    HERE / "phase1_kv_cache_minigunpoint_32token_ablations.py",
)
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)  # only defines functions/classes; no main() runs
abl.DEVICE = "cpu"  # force CPU timing regardless of MPS availability on this machine

REPEATS = 50
WARMUP = 5


def timed(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def stats(times: list[float]) -> dict:
    s = sorted(times)
    n = len(s)
    return {
        "mean_ms": sum(s) / n * 1000,
        "median_ms": s[n // 2] * 1000,
        "min_ms": s[0] * 1000,
        "max_ms": s[-1] * 1000,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--bridge", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()

    torch.set_num_threads(1)
    device = "cpu"
    torch.manual_seed(abl.SEED)

    tokenizer = abl.AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = abl.AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval().to(device)
    receiver = abl.AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval().to(device)
    for m in (sender, receiver):
        for p in m.parameters():
            p.requires_grad_(False)

    ckpt = torch.load(args.bridge, map_location=device, weights_only=False)
    sender_layer_selection = ckpt["sender_layer_selection"]
    n_layers = len(sender_layer_selection)
    adapter = abl.LowRankKVAdapter(n_layers, ckpt["head_dim"], ckpt["n_head"], ckpt["rank"], seed=abl.SEED)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.to(device)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)

    pool = abl.generate_pool(target_per_class=200)
    dev_docs = abl.select_docs(pool, "dev", abl.DEV_DOCS)
    dev_keys_set = sorted({"".join(s) for s, _ in dev_docs})
    assert dev_keys_set == ckpt["dev_docs_keys"], "dev split does not match the checkpointed training run"

    seq_ids = [abl.seq_to_ids(tokenizer, seq) for seq, _ in dev_docs]
    query_text = "Class:"
    full_texts = ["Sequence:" + "".join(f" {tok}" for tok in seq) + " " + query_text for seq, _ in dev_docs]
    full_ids = [tokenizer.encode(t, add_special_tokens=False) for t in full_texts]
    lens = {len(t) for t in full_ids}
    assert len(lens) == 1, f"re-prefill length not constant: {lens}"

    with torch.no_grad():
        def cache_transfer_path():
            ids = torch.tensor(seq_ids, dtype=torch.long, device=device)
            out = sender(ids, use_cache=True)
            pkv = out.past_key_values
            keys = [pkv.layers[i].keys for i in sender_layer_selection]
            values = [pkv.layers[i].values for i in sender_layer_selection]
            ak = adapter.forward_keys(keys)
            av = adapter.forward_values(values)
            dq = abl.query_ids(tokenizer, query_text, len(dev_docs))
            cache = abl.make_cache(ak, av)
            return abl.next_token_logits(receiver, dq, cache)

        def re_prefill_path():
            ids = torch.tensor(full_ids, dtype=torch.long, device=device)
            out = receiver(ids, use_cache=False)
            return out.logits[:, -1, :]

        # sanity: both paths return a real, finite logit tensor of the expected shape
        ct_logits = cache_transfer_path()
        rp_logits = re_prefill_path()
        assert ct_logits.shape == rp_logits.shape
        assert torch.isfinite(ct_logits).all() and torch.isfinite(rp_logits).all()

        ct_times = timed(cache_transfer_path, args.repeats, WARMUP)
        rp_times = timed(re_prefill_path, args.repeats, WARMUP)

    ct_stats = stats(ct_times)
    rp_stats = stats(rp_times)
    result = {
        "mechanism": "reprefill_efficiency_baseline",
        "method": (
            "CPU, single-threaded, PyTorch reference environment (not the native "
            "FASM/Accelerate path). Both paths process the same 32 original dev "
            "sequences in one batch, timed end-to-end from input tensor to output "
            "logits, after 5 warmup calls, over N repeats."
        ),
        "repeats": args.repeats,
        "batch_size": len(dev_docs),
        "sequence_length_tokens": len(seq_ids[0]),
        "reprefill_length_tokens": len(full_ids[0]),
        "cache_transfer_path": {
            "description": "sender prefill (12 layers) + adapter + receiver query-only forward",
            **ct_stats,
        },
        "receiver_reprefill_path": {
            "description": "receiver alone prefills the full raw sequence + query from scratch, no sender, no adapter",
            **rp_stats,
        },
        "reprefill_over_cache_transfer_speedup_median": rp_stats["median_ms"] / ct_stats["median_ms"],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
