#!/usr/bin/env python3
"""Scaling matrix for the Apple Silicon killer-workload product hypothesis:
"ultralight runtime for small static CPU models, where startup/RSS/deploy
footprint matter more than warm latency." tensor_killer_compare.py answered
this at exactly one size (~25k params). This answers it across a family of
increasingly large static MLPs (same architecture as the killer spike, only
hidden width grows) and finds where the hypothesis stops being compelling.

Does not touch the compiler, graph semantics, op traits, or executor —
read-only consumer of the canonical compile() path via
tensor_killer_scaling_native.c, exactly like every merged check before it.

=== Success criteria, fixed BEFORE any measurement in this run ===
Per size N, "niche survives" iff ALL THREE hold, comparing native against
the BEST of {onnxruntime, tinygrad} on each axis:
  1. cold startup   >= 5x  faster  (native_cold_ns <= best_cold_ns / 5)
  2. peak RSS       >= 3x  smaller (native_rss      <= best_rss      / 3)
  3. warm batch=1 latency no worse than 10x slower  (native_warm_ns <= best_warm_ns * 10)
These thresholds are the ones specified for this run and are not adjusted
after seeing results. Deploy footprint is measured and reported at every
size but is not part of the pass/fail gate above (informational — it
already established at n=1 that it isn't the binding constraint).
The crossover point is the smallest N where niche_survives becomes False
and stays False for all larger sizes measured.
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

COLD_MULT = 5.0
RSS_MULT = 3.0
WARM_MULT = 10.0

def parse(s):
    return dict(line.split('\t', 1) for line in s.strip().splitlines() if '\t' in line)

def tree_bytes(p):
    p = Path(p)
    return p.stat().st_size if p.is_file() else sum(x.stat().st_size for x in p.rglob('*') if x.is_file())

def run(cmd, env=None):
    return subprocess.run(cmd, text=True, capture_output=True, check=True, env=env).stdout

def measure_one(name, warm_cmd, cold_cmd, cold_reps):
    warm = parse(run(warm_cmd))
    cold = []
    for _ in range(cold_reps):
        e = os.environ.copy(); e['KILLER_SKIP_ACCURACY'] = '1'
        t = time.perf_counter_ns(); run(cold_cmd, e); cold.append(time.perf_counter_ns() - t)
    return {'warm_ns': float(warm['warm_median_ns']), 'accuracy': float(warm['accuracy']),
            'rss': int(warm['peak_rss_bytes']), 'checksum': float(warm['checksum']), 'cold_ns': statistics.median(cold)}

def one_size(a, hid, model_dir):
    subprocess.run([a.python, a.prep, 'prepare', '--mnist', a.mnist, '--out', str(model_dir), '--hid', str(hid), '--epochs', a.epochs],
                    check=True, capture_output=True, text=True)
    meta = json.loads((model_dir / 'meta.json').read_text())
    IN, HID, OUT = meta['IN'], meta['HID'], meta['OUT']
    m = model_dir
    common = [str(m / 'w1.bin'), str(m / 'b1.bin'), str(m / 'w2.bin'), str(m / 'b2.bin'), str(m / 'inputs.bin'), str(m / 'labels.bin')]
    cmds = {
        'native': [a.native, *common, str(a.reps), str(IN), str(HID), str(OUT)],
        'onnxruntime': [a.python, a.prep, 'onnxruntime', '--model', str(m), '--reps', str(a.reps)],
        'tinygrad': [a.python, a.prep, 'tinygrad', '--model', str(m), '--reps', str(a.reps)],
    }
    cold_cmds = {
        'native': [a.native, *common, '1', str(IN), str(HID), str(OUT)],
        'onnxruntime': [a.python, a.prep, 'onnxruntime', '--model', str(m), '--reps', '1'],
        'tinygrad': [a.python, a.prep, 'tinygrad', '--model', str(m), '--reps', '1'],
    }
    rows = {}
    for name in ('native', 'onnxruntime', 'tinygrad'):
        rows[name] = measure_one(name, cmds[name], cold_cmds[name], a.cold_reps)
    native_bytes = Path(a.native).stat().st_size + sum((m / n).stat().st_size for n in ('w1.bin', 'b1.bin', 'w2.bin', 'b2.bin'))
    rows['native']['deploy_bytes'] = native_bytes
    model_onnx_bytes = (m / 'model.onnx').stat().st_size
    for name, module in [('onnxruntime', 'onnxruntime'), ('tinygrad', 'tinygrad')]:
        loc = run([a.python, '-c', f'import {module};print({module}.__path__[0])']).strip()
        rows[name]['deploy_bytes'] = tree_bytes(loc) + Path(a.python).stat().st_size + model_onnx_bytes
    ok = (max(abs(rows['native']['checksum'] - rows[x]['checksum']) for x in ('onnxruntime', 'tinygrad'))
          <= 1e-2 * max(1, abs(rows['native']['checksum']))
          and len({round(v['accuracy'], 4) for v in rows.values()}) == 1)
    best_cold = min(rows['onnxruntime']['cold_ns'], rows['tinygrad']['cold_ns'])
    best_rss = min(rows['onnxruntime']['rss'], rows['tinygrad']['rss'])
    best_warm = min(rows['onnxruntime']['warm_ns'], rows['tinygrad']['warm_ns'])
    cold_ratio = best_cold / rows['native']['cold_ns']
    rss_ratio = best_rss / rows['native']['rss']
    warm_ratio = rows['native']['warm_ns'] / best_warm
    survives = ok and cold_ratio >= COLD_MULT and rss_ratio >= RSS_MULT and warm_ratio <= WARM_MULT
    return {'params': meta['params'], 'hid': HID, 'rows': rows, 'ok': ok,
            'cold_ratio': cold_ratio, 'rss_ratio': rss_ratio, 'warm_ratio': warm_ratio, 'survives': survives}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', required=True)
    ap.add_argument('--python', required=True)
    ap.add_argument('--prep', required=True, help='path to tensor_killer_scaling.py')
    ap.add_argument('--mnist', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sizes', default='32,126,630,1258,2516,5030',
                     help='comma-separated hidden widths; default hits ~25k/100k/500k/1M/2M/4M params at IN=784,OUT=10')
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--cold-reps', type=int, default=5)
    ap.add_argument('--epochs', default='20')
    a = ap.parse_args()
    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)

    print(f"criteria (fixed before measurement): niche survives at size N iff "
          f"cold_startup>={COLD_MULT:.0f}x faster AND peak_rss>={RSS_MULT:.0f}x smaller AND "
          f"warm_latency<={WARM_MULT:.0f}x slower, vs best of {{onnxruntime,tinygrad}}\n")
    print('params\thid\tengine\tcold_ns\twarm_ns\trss_bytes\tdeploy_bytes\taccuracy\tcold_x\trss_x\twarm_x\tniche_survives')

    results = []
    for hid in (int(s) for s in a.sizes.split(',')):
        model_dir = out_root / f'hid{hid}'
        try:
            res = one_size(a, hid, model_dir)
        except subprocess.CalledProcessError as e:
            print(f"# size hid={hid} FAILED: {e.stderr[-2000:] if e.stderr else e}", file=sys.stderr)
            continue
        results.append(res)
        for name, v in res['rows'].items():
            marker = f"{res['cold_ratio']:.1f}\t{res['rss_ratio']:.1f}\t{res['warm_ratio']:.1f}\t{res['survives']}" if name == 'native' else "\t\t\t"
            print(f"{res['params']}\t{res['hid']}\t{name}\t{v['cold_ns']:.0f}\t{v['warm_ns']:.0f}\t{v['rss']}\t{v['deploy_bytes']}\t{v['accuracy']:.4f}\t{marker}")
        if not res['ok']:
            print(f"# WARNING size hid={hid}: cross-engine correctness check FAILED (checksum/accuracy mismatch)", file=sys.stderr)

    crossover = None
    for res in results:
        if not res['survives']:
            crossover = res
            break
    print()
    if crossover is None and results:
        print(f"verdict\tNICHE SURVIVES across the entire measured range (up to {results[-1]['params']} params)")
    elif crossover is not None:
        prior = [r for r in results if r['params'] < crossover['params'] and r['survives']]
        floor = prior[-1]['params'] if prior else 0
        print(f"verdict\tCROSSOVER between {floor} and {crossover['params']} params "
              f"(cold_x={crossover['cold_ratio']:.1f} rss_x={crossover['rss_ratio']:.1f} warm_x={crossover['warm_ratio']:.1f} "
              f"at the failing point, thresholds were {COLD_MULT:.0f}x/{RSS_MULT:.0f}x/{WARM_MULT:.0f}x)")
    else:
        print("verdict\tNO DATA")

if __name__ == '__main__':
    main()
