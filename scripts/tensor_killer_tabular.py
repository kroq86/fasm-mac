#!/usr/bin/env python3
"""Prepare/run the tabular killer-workload model: a small 13->16->3 MLP on
the real UCI Wine dataset (178 samples, 13 physicochemical features, 3
cultivar classes — archive.ics.uci.edu/ml/machine-learning-databases/
wine/wine.data). Standard row-major (samples in class-grouped order in the
raw file, so this shuffles with a fixed seed before splitting)
zero-mean/unit-variance feature standardization fit on the train split
only. Same graph as every other MLP killer runner — MATMUL/BIAS_ADD/RELU/
MSE, unchanged; only the data and shapes (IN=13,HID=16,OUT=3,TEST_N=36)
differ, matching tensor_killer_tabular_native.c's compile-time constants.
"""
import argparse, os, resource, time
from pathlib import Path
import numpy as np

IN, HID, OUT = 13, 16, 3
TEST_N = 36

def load_wine(path):
    rows = np.loadtxt(path, delimiter=',')
    y = rows[:, 0].astype(np.int64) - 1  # 1,2,3 -> 0,1,2
    x = rows[:, 1:].astype(np.float32)
    rng = np.random.default_rng(11)
    order = rng.permutation(len(x))
    return x[order], y[order]

def prepare(a):
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    x, y = load_wine(a.wine)
    n_test = TEST_N
    ex, ey = x[:n_test], y[:n_test]
    tx, ty = x[n_test:], y[n_test:]
    mean, std = tx.mean(0), tx.std(0) + 1e-6
    tx = (tx - mean) / std
    ex = (ex - mean) / std

    rng = np.random.default_rng(7)
    w1 = (rng.standard_normal((IN, HID), dtype=np.float32) * float(1.0 / np.sqrt(IN))).astype(np.float32)
    b1 = np.zeros(HID, np.float32)
    w2 = (rng.standard_normal((HID, OUT), dtype=np.float32) * float(1.0 / np.sqrt(HID))).astype(np.float32)
    b2 = np.zeros(OUT, np.float32)
    lr = .05
    for _ in range(a.epochs):
        perm = rng.permutation(len(tx))
        for s in range(0, len(tx), 8):
            idx = perm[s:s + 8]
            xb, yb = tx[idx], ty[idx]
            z = xb @ w1 + b1; h = np.maximum(z, 0); pred = h @ w2 + b2
            y1h = np.eye(OUT, dtype=np.float32)[yb]
            g = 2 * (pred - y1h) / (len(xb) * OUT)
            dh = g @ w2.T; dz = dh * (z > 0)
            w2 -= lr * (h.T @ g); b2 -= lr * g.sum(0)
            w1 -= lr * (xb.T @ dz); b1 -= lr * dz.sum(0)

    for n, v in [('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2), ('inputs', ex.astype(np.float32))]:
        v.astype('<f4').tofile(o / f'{n}.bin')
    ey.astype('u1').tofile(o / 'labels.bin')

    import onnx
    from onnx import helper, TensorProto, numpy_helper
    nodes = [helper.make_node('MatMul', ['x', 'w1'], ['m1']), helper.make_node('Add', ['m1', 'b1'], ['a1']),
             helper.make_node('Relu', ['a1'], ['h']), helper.make_node('MatMul', ['h', 'w2'], ['m2']),
             helper.make_node('Add', ['m2', 'b2'], ['out'])]
    graph = helper.make_graph(nodes, 'tabular', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, IN])],
                               [helper.make_tensor_value_info('out', TensorProto.FLOAT, [1, OUT])],
                               [numpy_helper.from_array(w1, 'w1'), numpy_helper.from_array(b1, 'b1'),
                                numpy_helper.from_array(w2, 'w2'), numpy_helper.from_array(b2, 'b2')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=10)
    onnx.save(model, o / 'model.onnx')

def data(d):
    d = Path(d)
    return tuple(np.fromfile(d / f'{n}.bin', dtype='<f4').reshape(sh) for n, sh in
                 [('w1', (IN, HID)), ('b1', (HID,)), ('w2', (HID, OUT)), ('b2', (OUT,)), ('inputs', (TEST_N, IN))]) + (np.fromfile(d / 'labels.bin', dtype='u1'),)

def emit(engine, elapsed, correct, sink):
    print(f'engine\t{engine}\nwarm_median_ns\t{elapsed:.3f}\naccuracy\t{correct:.6f}\n'
          f'peak_rss_bytes\t{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}\nchecksum\t{sink:.9g}')

def ort(a):
    import onnxruntime as rt
    w1, b1, w2, b2, x, y = data(a.model)
    s = rt.InferenceSession(str(Path(a.model) / 'model.onnx'), providers=['CPUExecutionProvider'])
    name = s.get_inputs()[0].name
    xx = x[:1] if os.getenv('KILLER_SKIP_ACCURACY') else x
    yy = y[:len(xx)]
    out = np.concatenate([s.run(None, {name: q[None]})[0] for q in xx])
    correct = (out.argmax(1) == yy).mean()
    times = []
    for r in range(a.reps):
        t = time.perf_counter_ns(); z = s.run(None, {name: x[r % TEST_N:r % TEST_N + 1]})[0]; times.append(time.perf_counter_ns() - t)
    emit('onnxruntime', float(np.median(times)), correct, float(out.sum()))

def tiny(a):
    from tinygrad import Tensor
    w1, b1, w2, b2, x, y = data(a.model)
    tw1, tb1, tw2, tb2 = map(Tensor, (w1, b1, w2, b2))
    xx = x[:1] if os.getenv('KILLER_SKIP_ACCURACY') else x
    yy = y[:len(xx)]
    out = np.stack([((Tensor(q) @ tw1 + tb1).relu() @ tw2 + tb2).numpy() for q in xx])
    correct = (out.argmax(1) == yy).mean()
    times = []
    for r in range(a.reps):
        t = time.perf_counter_ns(); z = ((Tensor(x[r % TEST_N]) @ tw1 + tb1).relu() @ tw2 + tb2).numpy(); times.append(time.perf_counter_ns() - t)
    emit('tinygrad', float(np.median(times)), correct, float(out.sum()))

ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest='cmd', required=True)
p = sp.add_parser('prepare')
p.add_argument('--wine', required=True); p.add_argument('--out', required=True); p.add_argument('--epochs', type=int, default=200)
p.set_defaults(fn=prepare)
for n, fn in [('onnxruntime', ort), ('tinygrad', tiny)]:
    p = sp.add_parser(n); p.add_argument('--model', required=True); p.add_argument('--reps', type=int, default=1000)
    p.set_defaults(fn=fn)
a = ap.parse_args()
a.fn(a)
