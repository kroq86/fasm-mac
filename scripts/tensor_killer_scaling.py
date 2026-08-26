#!/usr/bin/env python3
"""Prepare/run one scaling-matrix model point. Generalizes
tensor_killer_mnist.py to arbitrary [IN,HID,OUT] MLP sizes: IN and OUT stay
fixed at MNIST's own 784/10 (so the same real test images/labels are used
at every point in the matrix), only HID varies, so total parameter count
scales as roughly HID*(IN+OUT+1)+OUT. Weight preparation happens once,
outside every measured path, identically to tensor_killer_mnist.py, and is
exported byte-for-byte to all three engines. Model quality (accuracy) is
NOT the point of this benchmark — deployment metrics are — so training is
kept light; the correctness gate that matters is that all three engines
agree with each other on the exact same weights, not that any of them
reaches a particular accuracy."""
import argparse, json, os, resource, struct, time
from pathlib import Path
import numpy as np

def idx(path, off):
    b = Path(path).read_bytes()
    shape = struct.unpack(">" + "I" * ((off - 4) // 4), b[4:off])
    return np.frombuffer(b, dtype=np.uint8, offset=off).copy().reshape(shape)

def prepare(a):
    d, o = Path(a.mnist), Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    IN, HID, OUT = 784, a.hid, 10
    tx = idx(d / 'train-images-idx3-ubyte', 16).reshape(-1, IN).astype(np.float32) / 255
    ty = idx(d / 'train-labels-idx1-ubyte', 8).reshape(-1)
    ex = idx(d / 't10k-images-idx3-ubyte', 16).reshape(-1, IN).astype(np.float32) / 255
    ey = idx(d / 't10k-labels-idx1-ubyte', 8).reshape(-1)
    rng = np.random.default_rng(7)
    # float(...) here matters: np.sqrt(IN) is a "strong" np.float64 scalar
    # under NEP 50, and float32_array * np.float64_scalar upcasts the whole
    # array to float64 — silently, no warning — which onnxruntime then
    # rejects (MatMul requires matching dtypes). float(...) forces a "weak"
    # Python float so the float32 array stays float32.
    w1 = rng.standard_normal((IN, HID), dtype=np.float32) * float(1.0 / np.sqrt(IN))
    b1 = np.zeros(HID, np.float32)
    w2 = rng.standard_normal((HID, OUT), dtype=np.float32) * float(1.0 / np.sqrt(HID))
    b2 = np.zeros(OUT, np.float32)
    lr = .15 / max(1.0, HID / 32.0)
    for _ in range(a.epochs):
        for s in range(0, 10000, 100):
            x = tx[s:s + 100]; y = np.eye(OUT, dtype=np.float32)[ty[s:s + 100]]
            z = x @ w1 + b1; h = np.maximum(z, 0); pred = h @ w2 + b2
            g = 2 * (pred - y) / (len(x) * OUT); dh = g @ w2.T; dz = dh * (z > 0)
            w2 -= lr * (h.T @ g); b2 -= lr * g.sum(0); w1 -= lr * (x.T @ dz); b1 -= lr * dz.sum(0)
    w1, b1, w2, b2 = (v.astype(np.float32) for v in (w1, b1, w2, b2))
    for n, v in [('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2), ('inputs', ex[:1000])]:
        v.astype('<f4').tofile(o / f'{n}.bin')
    ey[:1000].astype('u1').tofile(o / 'labels.bin')
    (o / 'meta.json').write_text(json.dumps({'IN': IN, 'HID': HID, 'OUT': OUT, 'params': IN * HID + HID + HID * OUT + OUT}))
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    nodes = [helper.make_node('MatMul', ['x', 'w1'], ['m1']), helper.make_node('Add', ['m1', 'b1'], ['a1']),
             helper.make_node('Relu', ['a1'], ['h']), helper.make_node('MatMul', ['h', 'w2'], ['m2']),
             helper.make_node('Add', ['m2', 'b2'], ['out'])]
    graph = helper.make_graph(nodes, 'mlp', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, IN])],
                               [helper.make_tensor_value_info('out', TensorProto.FLOAT, [1, OUT])],
                               [numpy_helper.from_array(w1, 'w1'), numpy_helper.from_array(b1, 'b1'),
                                numpy_helper.from_array(w2, 'w2'), numpy_helper.from_array(b2, 'b2')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=10)
    onnx.save(model, o / 'model.onnx')

def data(d):
    d = Path(d)
    meta = json.loads((d / 'meta.json').read_text())
    IN, HID, OUT = meta['IN'], meta['HID'], meta['OUT']
    arrs = tuple(np.fromfile(d / f'{n}.bin', dtype='<f4').reshape(sh) for n, sh in
                 [('w1', (IN, HID)), ('b1', (HID,)), ('w2', (HID, OUT)), ('b2', (OUT,)), ('inputs', (1000, IN))])
    return (meta,) + arrs + (np.fromfile(d / 'labels.bin', dtype='u1'),)

def emit(engine, elapsed, correct, sink):
    print(f'engine\t{engine}\nwarm_median_ns\t{elapsed:.3f}\naccuracy\t{correct:.6f}\n'
          f'peak_rss_bytes\t{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}\nchecksum\t{sink:.9g}')

def ort(a):
    import onnxruntime as rt
    meta, w1, b1, w2, b2, x, y = data(a.model)
    s = rt.InferenceSession(str(Path(a.model) / 'model.onnx'), providers=['CPUExecutionProvider'])
    name = s.get_inputs()[0].name
    xx = x[:1] if os.getenv('KILLER_SKIP_ACCURACY') else x
    yy = y[:len(xx)]
    out = np.concatenate([s.run(None, {name: q[None]})[0] for q in xx])
    correct = (out.argmax(1) == yy).mean()
    times = []
    for r in range(a.reps):
        t = time.perf_counter_ns(); z = s.run(None, {name: x[r % 1000:r % 1000 + 1]})[0]; times.append(time.perf_counter_ns() - t)
    emit('onnxruntime', float(np.median(times)), correct, float(out.sum()))

def tiny(a):
    from tinygrad import Tensor
    meta, w1, b1, w2, b2, x, y = data(a.model)
    tw1, tb1, tw2, tb2 = map(Tensor, (w1, b1, w2, b2))
    xx = x[:1] if os.getenv('KILLER_SKIP_ACCURACY') else x
    yy = y[:len(xx)]
    out = np.stack([((Tensor(q) @ tw1 + tb1).relu() @ tw2 + tb2).numpy() for q in xx])
    correct = (out.argmax(1) == yy).mean()
    times = []
    for r in range(a.reps):
        t = time.perf_counter_ns(); z = ((Tensor(x[r % 1000]) @ tw1 + tb1).relu() @ tw2 + tb2).numpy(); times.append(time.perf_counter_ns() - t)
    emit('tinygrad', float(np.median(times)), correct, float(out.sum()))

ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest='cmd', required=True)
p = sp.add_parser('prepare')
p.add_argument('--mnist', required=True); p.add_argument('--out', required=True)
p.add_argument('--hid', type=int, required=True); p.add_argument('--epochs', type=int, default=20)
p.set_defaults(fn=prepare)
for n, fn in [('onnxruntime', ort), ('tinygrad', tiny)]:
    p = sp.add_parser(n); p.add_argument('--model', required=True); p.add_argument('--reps', type=int, default=200)
    p.set_defaults(fn=fn)
a = ap.parse_args()
a.fn(a)
