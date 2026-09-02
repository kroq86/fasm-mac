#!/usr/bin/env python3
"""Reduces raw_algorithm.tsv to median/min/max per (ctxlen, mode). Reads
only; never edits the raw TSV. Run: python3 summarize.py raw_algorithm.tsv"""
import csv, sys
from statistics import median

path = sys.argv[1] if len(sys.argv) > 1 else "raw_algorithm.tsv"
rows = list(csv.DictReader(open(path), delimiter="\t"))
by_key = {}
max_abs_seen = 0.0
for r in rows:
    if "logits_max_abs_diff" in r and r["logits_max_abs_diff"] not in (None, ""):
        max_abs_seen = max(max_abs_seen, float(r["logits_max_abs_diff"]))
    if int(r["is_warmup"]):
        continue
    key = (int(r["ctxlen"]), r["mode"])
    by_key.setdefault(key, []).append(float(r["step_ns"]) / 1e6)

print(f'{"ctxlen":>7} {"mode":>8} {"n":>3} {"median_ms":>10} {"min_ms":>10} {"max_ms":>10}')
for (ctxlen, mode) in sorted(by_key.keys()):
    vals = by_key[(ctxlen, mode)]
    print(f"{ctxlen:>7} {mode:>8} {len(vals):>3} {median(vals):>10.2f} {min(vals):>10.2f} {max(vals):>10.2f}")
print(f"\nworst logits_max_abs_diff observed across all samples (warmup included): {max_abs_seen:.6g} (bound: 1e-3)")

print("\nratio (full_median / cached_median):")
for ctxlen in sorted(set(k[0] for k in by_key)):
    c = median(by_key[(ctxlen, "cached")])
    f = median(by_key[(ctxlen, "full")])
    print(f"  ctxlen={ctxlen}: cached={c:.2f}ms full={f:.2f}ms ratio={f/c:.2f}x")
