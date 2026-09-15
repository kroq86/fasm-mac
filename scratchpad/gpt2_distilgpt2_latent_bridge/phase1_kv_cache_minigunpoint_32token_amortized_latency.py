#!/usr/bin/env python3
"""Multi-query amortization latency check for the ordering adapter.

Motivation: the single-query re-prefill baseline (phase1_kv_cache_
minigunpoint_32token_latency_baseline.py) found cache transfer about 1.8x
slower than receiver re-prefill for ONE query -- the least favorable case
for cache transfer, since it pays the full sender cost for a single reuse.
The delegation architecture this line of work is motivated by (a strong
model processes context once, then hands cheaper models several narrower
sub-tasks) amortizes that sender cost over N downstream queries against the
SAME cache, not one. This script times the three components separately:

  (a) fixed cost: sender prefill (12 layers) + adapter application, paid once
  (b) marginal cost: one receiver query-only forward against the already
      adapted cache (no sender, no adapter), paid per downstream query
  (c) marginal cost: one full receiver re-prefill from raw text, paid per
      downstream query under the "just re-send the text" alternative

and computes the break-even query count N* at which cache transfer's total
cost (a + N*b) equals N*c, i.e. the number of reused queries needed before
amortizing the one-time sender cost pays for itself relative to re-prefilling
every time. CPU, single-threaded, PyTorch reference environment, matching
the single-query baseline's methodology; not the native FASM/Accelerate
path. All queries in this script are the SAME "Class:" query repeated N
times against the same cache (a stand-in for N distinct downstream
sub-tasks reusing the same processed context); it is a latency-structure
check, not a claim about what N distinct real queries would look like.
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
abl.DEVICE = "cpu"  # force CPU timing, matching the single-query latency baseline

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
    return {"mean_ms": sum(s) / n * 1000, "median_ms": s[n // 2] * 1000,
            "min_ms": s[0] * 1000, "max_ms": s[-1] * 1000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--bridge", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--threads", type=int, default=1, help="0 = leave PyTorch's default thread count")
    ap.add_argument("--batch-size", type=int, default=abl.DEV_DOCS, help="number of dev documents per timed call")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
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
    dev_docs_full = abl.select_docs(pool, "dev", abl.DEV_DOCS)
    dev_keys_set = sorted({"".join(s) for s, _ in dev_docs_full})
    assert dev_keys_set == ckpt["dev_docs_keys"], "dev split does not match the checkpointed training run"
    dev_docs = dev_docs_full[: args.batch_size]
    assert len(dev_docs) == args.batch_size

    seq_ids = [abl.seq_to_ids(tokenizer, seq) for seq, _ in dev_docs]
    query_text = "Class:"
    full_texts = ["Sequence:" + "".join(f" {tok}" for tok in seq) + " " + query_text for seq, _ in dev_docs]
    full_ids = [tokenizer.encode(t, add_special_tokens=False) for t in full_texts]
    assert len({len(t) for t in full_ids}) == 1

    with torch.no_grad():
        # (a) fixed cost: sender prefill + adapter, paid once, cache reused afterward
        def fixed_cost_path():
            ids = torch.tensor(seq_ids, dtype=torch.long, device=device)
            out = sender(ids, use_cache=True)
            pkv = out.past_key_values
            keys = [pkv.layers[i].keys for i in sender_layer_selection]
            values = [pkv.layers[i].values for i in sender_layer_selection]
            ak = adapter.forward_keys(keys)
            av = adapter.forward_values(values)
            return abl.make_cache(ak, av)

        # build one adapted cache to time the marginal per-query cost against
        ids = torch.tensor(seq_ids, dtype=torch.long, device=device)
        out = sender(ids, use_cache=True)
        pkv = out.past_key_values
        keys = [pkv.layers[i].keys for i in sender_layer_selection]
        values = [pkv.layers[i].values for i in sender_layer_selection]
        ak = adapter.forward_keys(keys)
        av = adapter.forward_values(values)

        def marginal_query_path():
            # DynamicCache is mutated by a forward call; rebuild it fresh each
            # timed call so repeated timing doesn't grow the cache across calls.
            cache = abl.make_cache(ak, av)
            dq = abl.query_ids(tokenizer, query_text, len(dev_docs))
            return abl.next_token_logits(receiver, dq, cache)

        def reprefill_path():
            full = torch.tensor(full_ids, dtype=torch.long, device=device)
            out = receiver(full, use_cache=False)
            return out.logits[:, -1, :]

        fixed_cost_path()  # sanity call
        marginal_query_path()
        reprefill_path()

        fixed_times = timed(fixed_cost_path, args.repeats, WARMUP)
        marginal_times = timed(marginal_query_path, args.repeats, WARMUP)
        reprefill_times = timed(reprefill_path, args.repeats, WARMUP)

    fixed_stats = stats(fixed_times)
    marginal_stats = stats(marginal_times)
    reprefill_stats = stats(reprefill_times)

    fixed_ms = fixed_stats["median_ms"]
    marginal_ms = marginal_stats["median_ms"]
    reprefill_ms = reprefill_stats["median_ms"]

    if reprefill_ms > marginal_ms:
        breakeven_n = fixed_ms / (reprefill_ms - marginal_ms)
    else:
        breakeven_n = None  # cache transfer never breaks even; per-query marginal cost already exceeds re-prefill

    totals_by_n = {}
    for n in (1, 2, 5, 10, 20, 50, 100):
        cache_total = fixed_ms + n * marginal_ms
        reprefill_total = n * reprefill_ms
        totals_by_n[str(n)] = {
            "cache_transfer_total_ms": cache_total,
            "reprefill_total_ms": reprefill_total,
            "cache_transfer_faster": cache_total < reprefill_total,
        }

    result = {
        "mechanism": "multi_query_amortized_latency",
        "method": (
            "CPU, single-threaded, PyTorch reference environment (not the native "
            "FASM/Accelerate path). Same 32 original dev sequences, one batch. "
            "Fixed cost = sender prefill + adapter, paid once. Marginal cache-query "
            "cost = one receiver query-only forward against the already-adapted "
            "cache. Marginal re-prefill cost = one full receiver forward over the "
            "raw sequence+query from scratch. Break-even N solves "
            "fixed + N*marginal_cache = N*marginal_reprefill."
        ),
        "repeats": args.repeats,
        "threads": args.threads if args.threads > 0 else "default",
        "batch_size": len(dev_docs),
        "fixed_cost_sender_plus_adapter": fixed_stats,
        "marginal_cost_cache_query": marginal_stats,
        "marginal_cost_reprefill": reprefill_stats,
        "breakeven_query_count": breakeven_n,
        "totals_by_query_count": totals_by_n,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
