#!/usr/bin/env python3
import math
import struct
import sys

from gunpoint_baseline import (
    BANDS, FEATURES, design, pooled_design, logistic_fit, logistic_accuracy, sigmoid,
)

HID = FEATURES


def load(path):
    with open(path, "rb") as f:
        magic, rows, bands, features, _ = struct.unpack("<8sIIII", f.read(24))
        if magic[:7] != b"GUNSEQ1" or bands != BANDS or features != FEATURES:
            raise ValueError("bad sequence fixture")
        records = []
        for _ in range(rows):
            is_test, rid = struct.unpack("<II", f.read(8))
            values = struct.unpack(f"<{bands * features}f", f.read(4 * bands * features))
            label, = struct.unpack("<f", f.read(4))
            record_bands = tuple(tuple(values[t * features:(t + 1) * features]) for t in range(bands))
            records.append((is_test, rid, record_bands, label))
    return records


def shuffled_order(rid):
    z = rid * 747796405 + 2891336453
    order = list(range(BANDS))
    for i in range(BANDS - 1, 0, -1):
        z = (z * 1664525 + 1013904223) & 0xffffffff
        j = z % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def ordered_order(rid):
    return tuple(range(BANDS))


def logistic_eval(train, test, order_fn):
    train_rows = [design(bands, order_fn(rid)) for _, rid, bands, _ in train]
    test_rows = [design(bands, order_fn(rid)) for _, rid, bands, _ in test]
    labels_train = [label for *_, label in train]
    labels_test = [label for *_, label in test]
    w = logistic_fit(train_rows, labels_train)
    return logistic_accuracy(test_rows, labels_test, w)


# Minimal hand-derived Elman RNN (tanh hidden state, BPTT) as a second,
# genuinely order-sensitive baseline distinct from the position-aware linear one.
def rnn_init(seed):
    state = [seed]

    def nxt():
        state[0] = (state[0] * 1664525 + 1013904223) & 0xffffffff
        return (state[0] / 0xffffffff - 0.5) * 0.4

    wx = [[nxt() for _ in range(FEATURES)] for _ in range(HID)]
    wh = [[nxt() for _ in range(HID)] for _ in range(HID)]
    bh = [0.0] * HID
    w_out = [nxt() for _ in range(HID)]
    return [wx, wh, bh, w_out, 0.0]


def rnn_forward(params, bands, order):
    wx, wh, bh, w_out, b_out = params
    h = [0.0] * HID
    hist = []
    for t in order:
        x = bands[t]
        h_prev = h
        a = [sum(wx[i][j] * x[j] for j in range(FEATURES)) +
             sum(wh[i][j] * h_prev[j] for j in range(HID)) + bh[i] for i in range(HID)]
        h = [math.tanh(v) for v in a]
        hist.append((x, h_prev, h))
    logit = sum(w_out[i] * h[i] for i in range(HID)) + b_out
    return logit, hist


def rnn_grad(params, bands, order, label):
    wx, wh, bh, w_out, b_out = params
    logit, hist = rnn_forward(params, bands, order)
    p = sigmoid(logit)
    dlogit = p - label
    h_final = hist[-1][2]
    dw_out = [dlogit * h_final[i] for i in range(HID)]
    db_out = dlogit
    dwx = [[0.0] * FEATURES for _ in range(HID)]
    dwh = [[0.0] * HID for _ in range(HID)]
    dbh = [0.0] * HID
    dh_next = [dlogit * w_out[i] for i in range(HID)]
    for x, h_prev, h in reversed(hist):
        da = [dh_next[i] * (1 - h[i] * h[i]) for i in range(HID)]
        for i in range(HID):
            for j in range(FEATURES):
                dwx[i][j] += da[i] * x[j]
            for j in range(HID):
                dwh[i][j] += da[i] * h_prev[j]
            dbh[i] += da[i]
        dh_next = [sum(da[k] * wh[k][i] for k in range(HID)) for i in range(HID)]
    return dwx, dwh, dbh, dw_out, db_out


def rnn_fit(train, order_fn, epochs=200, lr=0.3, seed=1234567):
    params = rnn_init(seed)
    wx, wh, bh, w_out, b_out = params
    n = len(train)
    for _ in range(epochs):
        gwx = [[0.0] * FEATURES for _ in range(HID)]
        gwh = [[0.0] * HID for _ in range(HID)]
        gbh = [0.0] * HID
        gw_out = [0.0] * HID
        gb_out = 0.0
        for _is_test, rid, bands, label in train:
            dwx, dwh, dbh, dw_out, db_out = rnn_grad(params, bands, order_fn(rid), label)
            for i in range(HID):
                for j in range(FEATURES):
                    gwx[i][j] += dwx[i][j]
                for j in range(HID):
                    gwh[i][j] += dwh[i][j]
                gbh[i] += dbh[i]
                gw_out[i] += dw_out[i]
            gb_out += db_out
        for i in range(HID):
            for j in range(FEATURES):
                wx[i][j] -= lr * gwx[i][j] / n
            for j in range(HID):
                wh[i][j] -= lr * gwh[i][j] / n
            bh[i] -= lr * gbh[i] / n
            w_out[i] -= lr * gw_out[i] / n
        params[4] = b_out - lr * gb_out / n
        b_out = params[4]
    return params


def rnn_accuracy(params, data, order_fn):
    correct = 0
    for _is_test, rid, bands, label in data:
        logit, _ = rnn_forward(params, bands, order_fn(rid))
        p = sigmoid(logit)
        correct += (p >= 0.5) == (label >= 0.5)
    return correct / len(data)


def main(path):
    records = load(path)
    train = [r for r in records if not r[0]]
    test = [r for r in records if r[0]]

    pooled_train = [pooled_design(bands) for _, _, bands, _ in train]
    pooled_test = [pooled_design(bands) for _, _, bands, _ in test]
    labels_train = [label for *_, label in train]
    labels_test = [label for *_, label in test]
    w_pool = logistic_fit(pooled_train, labels_train)
    acc_pooled = logistic_accuracy(pooled_test, labels_test, w_pool)
    acc_ordered = logistic_eval(train, test, ordered_order)
    acc_shuffled = logistic_eval(train, test, shuffled_order)
    print(f"gunpoint pooled-band logistic: train={len(train)} test={len(test)} "
          f"acc_pooled={acc_pooled:.3f} order_invariant=yes")
    print(f"gunpoint ordered-band logistic: acc_ordered={acc_ordered:.3f}")
    print(f"gunpoint shuffled-band logistic: acc_shuffled={acc_shuffled:.3f}")

    rnn_ordered = rnn_fit(train, ordered_order)
    acc_rnn_ordered = rnn_accuracy(rnn_ordered, test, ordered_order)
    rnn_shuffled = rnn_fit(train, shuffled_order)
    acc_rnn_shuffled = rnn_accuracy(rnn_shuffled, test, shuffled_order)
    print(f"gunpoint recurrent baseline: hidden={HID} acc_ordered={acc_rnn_ordered:.3f} "
          f"acc_shuffled={acc_rnn_shuffled:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
