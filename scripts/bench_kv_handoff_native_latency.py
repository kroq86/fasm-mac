#!/usr/bin/env python3
"""Native cache-transfer latency measurement for tensor-kv-handoff.

Times repeated invocations of the native ordering-checkpoint handoff binary,
via both the checked Python launcher (scripts/kv_handoff.py, which
SHA-256-verifies the full external bundle -- sender ~548MB, receiver ~328MB,
bridge, reference ~85MB, vocab, merges, cases -- on every call) and a direct
invocation of the binary itself (which still internally fingerprints sender,
receiver, and bridge per tensor_kv_handoff.c's own fail-closed loading
(argv[2]/argv[3]/argv[4] are SHA-256 hashes checked inside gpt2_load_weights_
layers/fingerprint, not filenames), but skips the launcher's redundant
re-hash of those same three files plus its unnecessary hash of
reference.safetensors/vocab.json/merges.txt/cases.txt, none of which the
--sequence code path reads).

This is a real, but one-sided, native-path latency number: it has no
counterpart "native receiver re-prefill" measurement, because no native CLI
exists that can run the 6-layer DistilGPT2 receiver on arbitrary raw text
(tensorctl gpt2 is compiled for the 12-layer GPT-2 shape only via a
compile-time #error guard; see NATIVE_HANDOFF.md and paper.md Sections 6-7).
This script measures one side of that comparison, not both, and does not
close that gap.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def timed(cmd: list[str], repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        times.append(time.perf_counter() - t0)
    return times


def stats(times: list[float]) -> dict:
    s = sorted(times)
    n = len(s)
    return {"mean_ms": sum(s) / n * 1000, "median_ms": s[n // 2] * 1000,
            "min_ms": s[0] * 1000, "max_ms": s[-1] * 1000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--binary", type=Path, default=ROOT / "fasm/build/out/tensor-kv-handoff")
    ap.add_argument("--sequence", default="A" * 16 + "B" * 16)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads((args.assets / "manifest.json").read_text())
    files = manifest["files"]

    launcher_cmd = [sys.executable, str(ROOT / "scripts/kv_handoff.py"),
                     "--assets", str(args.assets), "--binary", str(args.binary),
                     "--sequence", args.sequence]
    direct_cmd = [str(args.binary.resolve()), str(args.assets.resolve()),
                  files["sender.safetensors"], files["receiver.safetensors"], files["bridge.safetensors"],
                  "--sequence", args.sequence, "unused"]

    launcher_times = timed(launcher_cmd, args.repeats, args.warmup)
    direct_times = timed(direct_cmd, args.repeats, args.warmup)

    result = {
        "mechanism": "native_cache_transfer_latency",
        "method": (
            "Wall-clock timing of repeated native tensor-kv-handoff invocations "
            "(cold process each call, no server/persistent state). 'via_checked_"
            "launcher' times the full checked Python launcher (scripts/kv_handoff.py), "
            "which SHA-256-hashes the entire external bundle on every call. "
            "'via_direct_binary' times the native binary invoked directly with the "
            "manifest hashes as arguments, skipping the launcher's redundant "
            "re-hash but still subject to the binary's own internal fail-closed "
            "fingerprinting of sender/receiver/bridge. No native re-prefill "
            "counterpart exists to compare against."
        ),
        "sequence": args.sequence,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "bundle_sizes_bytes": {name: (args.assets / name).stat().st_size for name in files},
        "via_checked_launcher": stats(launcher_times),
        "via_direct_binary": stats(direct_times),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
