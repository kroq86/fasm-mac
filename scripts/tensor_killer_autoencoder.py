#!/usr/bin/env python3
"""Prepare/run the autoencoder killer-workload model: MLP autoencoder
(784->32->784) trained to reconstruct only digit '0' from real MNIST,
evaluated as anomaly detection (0=normal, non-0=anomaly) on a fixed
TEST_N-sample mixed test set. No external threshold artifact — matching
tensor_killer_autoencoder_native.c, the decision boundary is the *median*
reconstruction MSE over that same fixed test set, computed independently
by each engine from the same weights and data, so it can't drift out of
sync between engines without an extra shared file. Same graph as every
other MLP killer runner (MATMUL/BIAS_ADD/RELU, unchanged) with OUT=784
instead of 10 and no final loss node — this file is pure inference plus
threshold-free evaluation, not training-time loss.
"""
import argparse, os, resource, struct, time
from pathlib import Path
import numpy as np

IN, HID, OUT = 784, 64, 784
TEST_N = 200

def idx(path, off):
    b = Path(path).read_bytes()
    shape = struct.unpack(">" + "I" * ((off - 4) // 4), b[4:off])
    return np.frombuffer(b, dtype=np.uint8, offset=off).copy().reshape(shape)

def prepare(a):
    d, o = Path(a.mnist), Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    tx = idx(d / 'train-images-idx3-ubyte', 16).reshape(-1, IN).astype(np.float32) / 255
    ty = idx(d / 'train-labels-idx1-ubyte', 8).reshape(-1)
    ex = idx(d / 't10k-images-idx3-ubyte', 16).reshape(-1, IN).astype(np.float32) / 255
    ey = idx(d / 't10k-labels-idx1-ubyte', 8).reshape(-1)

    zeros_train = tx[ty == 0][:800]  # normal-only training set
    rng = np.random.default_rng(7)
    w1 = (rng.standard_normal((IN, HID), dtype=np.float32) * float(1.0 / np.sqrt(IN))).astype(np.float32)
    b1 = np.zeros(HID, np.float32)
    w2 = (rng.standard_normal((HID, OUT), dtype=np.float32) * float(1.0 / np.sqrt(HID))).astype(np.float32)
    b2 = np.zeros(OUT, np.float32)
    lr = .1
    for _ in range(a.epochs):
        perm = rng.permutation(len(zeros_train))
        for s in range(0, len(zeros_train), 40):
            xb = zeros_train[perm[s:s + 40]]
            z = xb @ w1 + b1; h = np.maximum(z, 0); pred = h @ w2 + b2
            g = 2 * (pred - xb) / (len(xb) * OUT)  # reconstruction target IS the input
            dh = g @ w2.T; dz = dh * (z > 0)
            w2 -= lr * (h.T @ g); b2 -= lr * g.sum(0)
            w1 -= lr * (xb.T @ dz); b1 -= lr * dz.sum(0)

    # fixed mixed test set: TEST_N/2 real '0's (normal) + TEST_N/2 random
    # non-'0' digits (anomalies), held out from training
    half = TEST_N // 2
    ez = ex[ey == 0][:half]
    enz_idx = np.flatnonzero(ey != 0)[:half]
    enz = ex[enz_idx]
    test_x = np.concatenate([ez, enz], axis=0)
    test_y = np.concatenate([np.zeros(len(ez), np.uint8), np.ones(len(enz), np.uint8)])

    for n, v in [('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2), ('inputs', test_x.astype(np.float32))]:
        v.astype('<f4').tofile(o / f'{n}.bin')
    test_y.astype('u1').tofile(o / 'labels.bin')

    import onnx
    from onnx import helper, TensorProto, numpy_helper
    nodes = [helper.make_node('MatMul', ['x', 'w1'], ['m1']), helper.make_node('Add', ['m1', 'b1'], ['a1']),
             helper.make_node('Relu', ['a1'], ['h']), helper.make_node('MatMul', ['h', 'w2'], ['m2']),
             helper.make_node('Add', ['m2', 'b2'], ['out'])]
    graph = helper.make_graph(nodes, 'autoencoder', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, IN])],
                               [helper.make_tensor_value_info('out', TensorProto.FLOAT, [1, OUT])],
                               [numpy_helper.from_array(w1, 'w1'), numpy_helper.from_array(b1, 'b1'),
                                numpy_helper.from_array(w2, 'w2'), numpy_helper.from_array(b2, 'b2')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=10)
    onnx.save(model, o / 'model.onnx')

def data(d):
    d = Path(d)
    return tuple(np.fromfile(d / f'{n}.bin', dtype='<f4').reshape(sh) for n, sh in
                 [('w1', (IN, HID)), ('b1', (HID,)), ('w2', (HID, OUT)), ('b2', (OUT,)), ('inputs', (TEST_N, IN))]) + (np.fromfile(d / 'labels.bin', dtype='u1'),)

def anomaly_accuracy(recon, x, y):
    mse = ((recon - x) ** 2).mean(1)
    # mean, not median — see tensor_killer_autoencoder_native.c's comment:
    # the median is always exactly one sample's own value, guaranteeing a
    # knife-edge tie that flips under float-summation-order differences
    # between engines; the mean essentially never is.
    threshold = mse.mean()
    pred = (mse > threshold).astype(np.uint8)
    return (pred == y).mean()

def emit(engine, elapsed, correct, sink):
    print(f'engine\t{engine}\nwarm_median_ns\t{elapsed:.3f}\naccuracy\t{correct:.6f}\n'
          f'peak_rss_bytes\t{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}\nchecksum\t{sink:.9g}')

def ort(a):
    import onnxruntime as rt
    w1, b1, w2, b2, x, y = data(a.model)
    s = rt.InferenceSession(str(Path(a.model) / 'model.onnx'), providers=['CPUExecutionProvider'])
    name = s.get_inputs()[0].name
    if os.getenv('KILLER_SKIP_ACCURACY'):
        out = np.concatenate([s.run(None, {name: x[:1]})[0]])
        correct = 0.0
    else:
        out = np.concatenate([s.run(None, {name: q[None]})[0] for q in x])
        correct = anomaly_accuracy(out, x, y)
    times = []
    for r in range(a.reps):
        t = time.perf_counter_ns(); z = s.run(None, {name: x[r % TEST_N:r % TEST_N + 1]})[0]; times.append(time.perf_counter_ns() - t)
    emit('onnxruntime', float(np.median(times)), correct, float(out.sum()))

def tiny(a):
    from tinygrad import Tensor
    w1, b1, w2, b2, x, y = data(a.model)
    tw1, tb1, tw2, tb2 = map(Tensor, (w1, b1, w2, b2))
    def fwd(q): return ((Tensor(q) @ tw1 + tb1).relu() @ tw2 + tb2).numpy()
    if os.getenv('KILLER_SKIP_ACCURACY'):
        out = np.stack([fwd(x[0])])
        correct = 0.0
    else:
        out = np.stack([fwd(q) for q in x])
        correct = anomaly_accuracy(out, x, y)
    times = []
    for r in range(a.reps):
        t = time.perf_counter_ns(); z = fwd(x[r % TEST_N]); times.append(time.perf_counter_ns() - t)
    emit('tinygrad', float(np.median(times)), correct, float(out.sum()))

ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest='cmd', required=True)
p = sp.add_parser('prepare')
p.add_argument('--mnist', required=True); p.add_argument('--out', required=True); p.add_argument('--epochs', type=int, default=150)
p.set_defaults(fn=prepare)
for n, fn in [('onnxruntime', ort), ('tinygrad', tiny)]:
    p = sp.add_parser(n); p.add_argument('--model', required=True); p.add_argument('--reps', type=int, default=1000)
    p.set_defaults(fn=fn)
a = ap.parse_args()
a.fn(a)
