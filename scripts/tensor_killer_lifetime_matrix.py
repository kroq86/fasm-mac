#!/usr/bin/env python3
"""Lifetime-crossover benchmark for the killer-workload's ~25k-param model
(the same 784->32->10 MLP tensor_killer_compare.py already measures cold-
startup and warm latency for in isolation). Neither of those two numbers
alone describes a workload shaped "start a process, do N inferences,
exit": cold startup ignores the N inferences; warm latency (steady state,
after 100+ warm-up calls) discards startup entirely. This measures what
actually matters for that shape directly: total wall-clock time of a
fresh process that starts, loads the model, and performs exactly N
inferences, for N in {1,10,100,1000,10000} — not the linear back-of-
envelope estimate (cold + N*warm), the real thing, spawning an actual
process each time.

No new native or Python inference code: tensor_killer_mnist_native.c
already takes `reps` as argv[7] and tensor_killer_mnist.py's onnxruntime/
tinygrad subcommands already take --reps — this only varies that
parameter and times the whole subprocess externally, exactly like
tensor_killer_compare.py's existing cold-timing methodology already does
at reps=1.
"""
import argparse, os, statistics, subprocess, time
from pathlib import Path

# fewer repeats at large N to keep total wall time sane — tinygrad's
# per-call overhead in this benchmark (~3ms, no @TinyJit caching, matching
# how tensor_killer_mnist.py already calls it) makes N=10000 the slow case
REPEATS_BY_N = {1: 7, 10: 7, 100: 5, 1000: 3, 10000: 2}

def run_timed(cmd, env, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        times.append(time.perf_counter_ns() - t0)
    return statistics.median(times)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', required=True)
    ap.add_argument('--python', required=True)
    ap.add_argument('--runner', required=True, help='path to tensor_killer_mnist.py')
    ap.add_argument('--model', required=True)
    ap.add_argument('--ns', default='1,10,100,1000,10000')
    a = ap.parse_args()
    m = Path(a.model)
    common = [str(m / 'w1.bin'), str(m / 'b1.bin'), str(m / 'w2.bin'), str(m / 'b2.bin'),
              str(m / 'inputs.bin'), str(m / 'labels.bin')]
    env = os.environ.copy()
    env['KILLER_SKIP_ACCURACY'] = '1'

    ns = [int(x) for x in a.ns.split(',')]
    rows = {}
    for n in ns:
        cmds = {
            'native': [a.native, *common, str(n)],
            'onnxruntime': [a.python, a.runner, 'onnxruntime', '--model', str(m), '--reps', str(n)],
            'tinygrad': [a.python, a.runner, 'tinygrad', '--model', str(m), '--reps', str(n)],
        }
        repeats = REPEATS_BY_N.get(n, 3)
        rows[n] = {}
        for name, cmd in cmds.items():
            rows[n][name] = run_timed(cmd, env, repeats)

    print('N_inferences_per_process\tnative_total_ns\tonnxruntime_total_ns\ttinygrad_total_ns\twinner')
    for n in ns:
        r = rows[n]
        winner = min(r, key=r.get)
        print(f"{n}\t{r['native']:.0f}\t{r['onnxruntime']:.0f}\t{r['tinygrad']:.0f}\t{winner}")

    crossover_lo, crossover_hi = None, None
    for i, n in enumerate(ns):
        if rows[n]['onnxruntime'] < rows[n]['native']:
            crossover_hi = n
            crossover_lo = ns[i - 1] if i > 0 else 0
            break
    print()
    if crossover_hi is None:
        print(f"verdict\tnative wins total wall-clock time at every measured N up to {ns[-1]} inferences/process lifetime")
    else:
        print(f"verdict\tempirical crossover for total wall-clock time is between N={crossover_lo} and N={crossover_hi} inferences per process lifetime "
              f"(onnxruntime's amortized startup cost stops dominating somewhere in that range)")

if __name__ == '__main__':
    main()
