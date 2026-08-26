#!/usr/bin/env python3
"""Prepare/run the CNN killer-workload model: conv(5x5,4 filters, valid)
-> relu -> 2x2 max pool -> fc(576->10), real MNIST, matching the exact
architecture tensor_merged_cnn_pool_mnist_check.c already trains through
the canonical compiler. Weight preparation (batched im2col conv, vectorized
via numpy's sliding_window_view) happens once outside every measured path,
exported byte-for-byte to native/onnxruntime/tinygrad, same convention as
tensor_killer_mnist.py.

File layout, matching tensor_killer_mnist_native.c's CLI so
tensor_killer_compare.py / tensor_killer_lifetime_matrix.py work
unmodified: w1=conv weight [COUT,CIN*KH*KW], b1=unused placeholder (this
design has no conv bias, matching the canonical CONV kernel), w2=fc weight
[POOL_OUT,OUT], b2=real fc bias [OUT].
"""
import argparse, json, os, resource, struct, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HIN, WIN, CIN, COUT, KH, KW = 28, 28, 1, 4, 5, 5
HOUT, WOUT = HIN - KH + 1, WIN - KW + 1
PPH, PPW = 2, 2
PHOUT, PWOUT = HOUT // PPH, WOUT // PPW
POOL_OUT = PHOUT * PWOUT * COUT
OUT = 10

def idx(path, off):
    b = Path(path).read_bytes()
    shape = struct.unpack(">" + "I" * ((off - 4) // 4), b[4:off])
    return np.frombuffer(b, dtype=np.uint8, offset=off).copy().reshape(shape)

def im2col_batch(x):
    # x: (B, HIN, WIN) -> (B, HOUT, WOUT, KH, KW)
    return sliding_window_view(x, (KH, KW), axis=(1, 2))

def conv_forward(x, w):
    # x: (B,HIN,WIN) w: (COUT, CIN*KH*KW) -> conv:(B,HOUT,WOUT,COUT), patches:(B,HOUT,WOUT,KH*KW)
    win = im2col_batch(x)  # (B,HOUT,WOUT,KH,KW)
    patches = win.reshape(win.shape[0], HOUT, WOUT, KH * KW)
    conv = patches @ w.T  # (B,HOUT,WOUT,COUT)
    return conv, patches

def pool_forward(h):
    # h: (B,HOUT,WOUT,COUT) -> pooled:(B,PHOUT,PWOUT,COUT), argmax mask same shape as h (bool)
    b = h.shape[0]
    blocks = h.reshape(b, PHOUT, PPH, PWOUT, PPW, COUT)
    pooled = blocks.max(axis=(2, 4))
    # exact ties (extremely unlikely with random floats) would double-route
    # gradient into more than one input; harmless for training-prep-only use
    mask = blocks == pooled[:, :, None, :, None, :]
    return pooled.reshape(b, POOL_OUT), mask

def prepare(a):
    d, o = Path(a.mnist), Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    tx = idx(d / 'train-images-idx3-ubyte', 16).astype(np.float32) / 255
    ty = idx(d / 'train-labels-idx1-ubyte', 8).reshape(-1)
    ex = idx(d / 't10k-images-idx3-ubyte', 16).astype(np.float32) / 255
    ey = idx(d / 't10k-labels-idx1-ubyte', 8).reshape(-1)

    rng = np.random.default_rng(7)
    cw = (rng.standard_normal((COUT, CIN * KH * KW), dtype=np.float32) * float(1.0 / np.sqrt(KH * KW))).astype(np.float32)
    fcw = (rng.standard_normal((POOL_OUT, OUT), dtype=np.float32) * float(1.0 / np.sqrt(POOL_OUT))).astype(np.float32)
    fcb = np.zeros(OUT, np.float32)

    train_n = min(3000, len(tx))
    tx, ty = tx[:train_n], ty[:train_n]
    lr = .05
    batch = 50
    for _ in range(a.epochs):
        for s in range(0, train_n, batch):
            xb, yb = tx[s:s + batch], ty[s:s + batch]
            conv, patches = conv_forward(xb, cw)  # (b,HOUT,WOUT,COUT)
            relu = np.maximum(conv, 0)
            pooled, mask = pool_forward(relu)  # (b,POOL_OUT)
            pred = pooled @ fcw + fcb
            y1h = np.eye(OUT, dtype=np.float32)[yb]
            g = 2 * (pred - y1h) / (len(xb) * OUT)
            dfcw = pooled.T @ g
            dfcb = g.sum(0)
            dpooled = g @ fcw.T  # (b,POOL_OUT)
            dpooled_full = dpooled.reshape(len(xb), PHOUT, PWOUT, COUT)
            drelu = np.zeros_like(relu).reshape(len(xb), PHOUT, PPH, PWOUT, PPW, COUT)
            drelu[mask] = np.broadcast_to(dpooled_full[:, :, None, :, None, :], mask.shape)[mask]
            drelu = drelu.reshape(len(xb), HOUT, WOUT, COUT)
            dconv = drelu * (conv > 0)
            dcw = np.einsum('bhwk,bhwc->ck', patches.reshape(len(xb), HOUT, WOUT, KH * KW), dconv).astype(np.float32)
            cw -= lr * dcw
            fcw -= lr * dfcw.astype(np.float32)
            fcb -= lr * dfcb.astype(np.float32)

    b1_unused = np.zeros(1, np.float32)
    for n, v in [('w1', cw.reshape(-1)), ('b1', b1_unused), ('w2', fcw), ('b2', fcb), ('inputs', ex[:1000].reshape(1000, -1))]:
        v.astype('<f4').tofile(o / f'{n}.bin')
    ey[:1000].astype('u1').tofile(o / 'labels.bin')
    (o / 'meta.json').write_text(json.dumps({'HIN': HIN, 'WIN': WIN, 'CIN': CIN, 'COUT': COUT, 'KH': KH, 'KW': KW,
                                              'POOL_OUT': POOL_OUT, 'OUT': OUT,
                                              'params': COUT * CIN * KH * KW + POOL_OUT * OUT + OUT}))

    import onnx
    from onnx import helper, TensorProto, numpy_helper
    conv_w = cw.reshape(COUT, CIN, KH, KW)
    nodes = [
        helper.make_node('Conv', ['x', 'cw'], ['c'], kernel_shape=[KH, KW]),
        helper.make_node('Relu', ['c'], ['h']),
        helper.make_node('MaxPool', ['h'], ['p'], kernel_shape=[PPH, PPW], strides=[PPH, PPW]),
        # ONNX MaxPool is NCHW; the canonical CONV/POOL kernels this model
        # is compared against flatten HWC ((oh,ow,co) order — documented in
        # tensor_semantic_compiler.h). Transpose before Flatten so both
        # sides feed the FC weight the same feature order.
        helper.make_node('Transpose', ['p'], ['pt'], perm=[0, 2, 3, 1]),
        helper.make_node('Flatten', ['pt'], ['pf'], axis=1),
        helper.make_node('MatMul', ['pf', 'w2'], ['m2']),
        helper.make_node('Add', ['m2', 'b2'], ['out']),
    ]
    graph = helper.make_graph(nodes, 'cnn', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, CIN, HIN, WIN])],
                               [helper.make_tensor_value_info('out', TensorProto.FLOAT, [1, OUT])],
                               [numpy_helper.from_array(conv_w, 'cw'), numpy_helper.from_array(fcw, 'w2'),
                                numpy_helper.from_array(fcb, 'b2')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=10)
    onnx.save(model, o / 'model.onnx')

def data(d):
    d = Path(d)
    meta = json.loads((d / 'meta.json').read_text())
    cw = np.fromfile(d / 'w1.bin', dtype='<f4').reshape(COUT, CIN * KH * KW)
    fcw = np.fromfile(d / 'w2.bin', dtype='<f4').reshape(POOL_OUT, OUT)
    fcb = np.fromfile(d / 'b2.bin', dtype='<f4')
    x = np.fromfile(d / 'inputs.bin', dtype='<f4').reshape(1000, HIN, WIN)
    y = np.fromfile(d / 'labels.bin', dtype='u1')
    return meta, cw, fcw, fcb, x, y

def emit(engine, elapsed, correct, sink):
    print(f'engine\t{engine}\nwarm_median_ns\t{elapsed:.3f}\naccuracy\t{correct:.6f}\n'
          f'peak_rss_bytes\t{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}\nchecksum\t{sink:.9g}')

def ort(a):
    import onnxruntime as rt
    meta, cw, fcw, fcb, x, y = data(a.model)
    s = rt.InferenceSession(str(Path(a.model) / 'model.onnx'), providers=['CPUExecutionProvider'])
    name = s.get_inputs()[0].name
    xx = x[:1] if os.getenv('KILLER_SKIP_ACCURACY') else x
    yy = y[:len(xx)]
    out = np.concatenate([s.run(None, {name: q[None, None]})[0] for q in xx])
    correct = (out.argmax(1) == yy).mean()
    times = []
    for r in range(a.reps):
        t = time.perf_counter_ns(); z = s.run(None, {name: x[r % 1000][None, None]})[0]; times.append(time.perf_counter_ns() - t)
    emit('onnxruntime', float(np.median(times)), correct, float(out.sum()))

def tiny(a):
    from tinygrad import Tensor
    meta, cw, fcw, fcb, x, y = data(a.model)
    tcw, tfcw, tfcb = Tensor(cw.reshape(COUT, CIN, KH, KW)), Tensor(fcw), Tensor(fcb)
    def fwd(img):
        t = Tensor(img.reshape(1, CIN, HIN, WIN))
        c = t.conv2d(tcw).relu().max_pool2d(kernel_size=(PPH, PPW), stride=(PPH, PPW))
        # NCHW -> HWC to match native's (oh,ow,co) flatten order, same
        # reason as the ONNX Transpose above
        c = c.permute(0, 2, 3, 1)
        return (c.reshape(1, POOL_OUT) @ tfcw + tfcb).numpy()[0]
    xx = x[:1] if os.getenv('KILLER_SKIP_ACCURACY') else x
    yy = y[:len(xx)]
    out = np.stack([fwd(q) for q in xx])
    correct = (out.argmax(1) == yy).mean()
    times = []
    for r in range(a.reps):
        t = time.perf_counter_ns(); z = fwd(x[r % 1000]); times.append(time.perf_counter_ns() - t)
    emit('tinygrad', float(np.median(times)), correct, float(out.sum()))

ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest='cmd', required=True)
p = sp.add_parser('prepare')
p.add_argument('--mnist', required=True); p.add_argument('--out', required=True); p.add_argument('--epochs', type=int, default=8)
p.set_defaults(fn=prepare)
for n, fn in [('onnxruntime', ort), ('tinygrad', tiny)]:
    p = sp.add_parser(n); p.add_argument('--model', required=True); p.add_argument('--reps', type=int, default=1000)
    p.set_defaults(fn=fn)
a = ap.parse_args()
a.fn(a)
