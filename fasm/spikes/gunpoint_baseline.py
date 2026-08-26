#!/usr/bin/env python3
"""Dependency-free baseline for the UCR GunPoint time-series benchmark.

GunPoint separates "Gun" vs "Point" hand motions from a length-150
univariate trace; unlike NIR-MFCO's false-colour bands, order here is a
physical timeline, not an interchangeable label. The archive stays outside
the repo: point --train/--test, or a directory containing the classic
TRAIN/TEST files, at your own extracted copy.
"""

import argparse
import math
import os
import struct
import sys

LENGTH = 150
TRAIN_ROWS = 50
TEST_ROWS = 150
BANDS = 3
FEATURES = 4  # mean, std, min, max per band


def parse_rows(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            fields = line.replace(",", " ").split()
            if not fields:
                continue
            label = float(fields[0])
            values = [float(v) for v in fields[1:]]
            if len(values) != LENGTH:
                raise ValueError(f"{path}: expected {LENGTH} values, found {len(values)}")
            rows.append((label, values))
    return rows


def find_split(directory):
    train = test = None
    for name in sorted(os.listdir(directory)):
        upper = name.upper()
        if "TRAIN" in upper:
            train = os.path.join(directory, name)
        elif "TEST" in upper:
            test = os.path.join(directory, name)
    if not train or not test:
        raise ValueError(f"{directory}: could not find TRAIN/TEST files")
    return train, test


def zscore(values):
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    sd = math.sqrt(var) or 1.0
    return [(v - mean) / sd for v in values]


def band_features(values):
    span = LENGTH // BANDS
    bands = []
    for t in range(BANDS):
        chunk = values[t * span:(t + 1) * span]
        mean = sum(chunk) / span
        var = sum((v - mean) ** 2 for v in chunk) / span
        bands.append((mean, math.sqrt(var), min(chunk), max(chunk)))
    return tuple(bands)


def design(bands, order):
    row = [1.0]
    for t in order:
        row.extend(bands[t])
    return row


def pooled_design(bands):
    row = [sum(band[j] for band in bands) / len(bands) for j in range(FEATURES)]
    return [1.0] + row


def sigmoid(z):
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def logistic_fit(rows, labels, epochs=400, lr=0.5):
    width = len(rows[0])
    w = [0.0] * width
    n = len(rows)
    for _ in range(epochs):
        grad = [0.0] * width
        for row, y in zip(rows, labels):
            p = sigmoid(sum(x * wi for x, wi in zip(row, w)))
            err = p - y
            for i, x in enumerate(row):
                grad[i] += err * x
        for i in range(width):
            w[i] -= lr * grad[i] / n
    return w


def logistic_accuracy(rows, labels, w):
    correct = 0
    for row, y in zip(rows, labels):
        p = sigmoid(sum(x * wi for x, wi in zip(row, w)))
        correct += (p >= 0.5) == (y >= 0.5)
    return correct / len(rows)


def load_records(train_path, test_path):
    raw_train = parse_rows(train_path)
    raw_test = parse_rows(test_path)
    if len(raw_train) != TRAIN_ROWS:
        raise ValueError(f"expected {TRAIN_ROWS} train rows, found {len(raw_train)}")
    if len(raw_test) != TEST_ROWS:
        raise ValueError(f"expected {TEST_ROWS} test rows, found {len(raw_test)}")
    labels = sorted({label for label, _ in raw_train} | {label for label, _ in raw_test})
    if len(labels) != 2:
        raise ValueError(f"expected 2 classes, found {labels}")
    label_map = {labels[0]: 0.0, labels[1]: 1.0}
    records = []
    rid = 0
    for is_test, rows in ((0, raw_train), (1, raw_test)):
        for label, values in rows:
            bands = band_features(zscore(values))
            records.append((is_test, rid, bands, label_map[label]))
            rid += 1
    return records


def evaluate(records):
    train = [r for r in records if not r[0]]
    test = [r for r in records if r[0]]
    y_train = [r[3] for r in train]
    y_test = [r[3] for r in test]
    order = tuple(range(BANDS))
    w_ordered = logistic_fit([design(r[2], order) for r in train], y_train)
    w_pooled = logistic_fit([pooled_design(r[2]) for r in train], y_train)
    acc_ordered = logistic_accuracy([design(r[2], order) for r in test], y_test, w_ordered)
    acc_pooled = logistic_accuracy([pooled_design(r[2]) for r in test], y_test, w_pooled)
    return len(train), len(test), acc_ordered, acc_pooled


def export_sequence_fixture(path, records):
    """Write a small, stable float32 hand-off file for the x86 tensor spike."""
    with open(path, "wb") as output:
        output.write(struct.pack("<8sIIII", b"GUNSEQ1\0", len(records), BANDS, FEATURES, 0))
        for is_test, rid, bands, label in records:
            output.write(struct.pack("<II", is_test, rid))
            for band in bands:
                output.write(struct.pack("<4f", *band))
            output.write(struct.pack("<f", label))


def self_test():
    # Same {v0, v1, v2} band means for both classes, only the order flips:
    # pooling must sit at chance while an order-aware model is trivial.
    records = []
    for rid in range(80):
        lo = 0.1 * (rid % 5)
        means = (lo, lo + 0.5, lo + 1.0)
        ascending = rid % 2 == 0
        ordered_means = means if ascending else tuple(reversed(means))
        bands = tuple((m, 0.02, m - 0.02, m + 0.02) for m in ordered_means)
        is_test = 1 if rid >= 40 else 0
        records.append((is_test, rid, bands, 1.0 if ascending else 0.0))
    train, test, acc_ordered, acc_pooled = evaluate(records)
    if train != 40 or test != 40 or acc_ordered < 0.95 or acc_pooled > 0.65:
        raise AssertionError(
            f"gunpoint baseline self-test failed: acc_ordered={acc_ordered:.3f} acc_pooled={acc_pooled:.3f}")
    print(f"gunpoint baseline self-test passed: train=40 test=40 "
          f"rule=ascending-vs-descending-band-means acc_ordered={acc_ordered:.3f} acc_pooled={acc_pooled:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", help="directory with UCR GunPoint TRAIN/TEST files")
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--export-sequence")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.train and args.test:
        train_path, test_path = args.train, args.test
    elif args.directory:
        train_path, test_path = find_split(args.directory)
    else:
        parser.error("directory or --train/--test is required unless --self-test is used")
    records = load_records(train_path, test_path)
    if args.export_sequence:
        export_sequence_fixture(args.export_sequence, records)
    train, test, acc_ordered, acc_pooled = evaluate(records)
    print(f"gunpoint baseline samples={len(records)} train={train} test={test} "
          f"bands={BANDS} features={FEATURES} acc_ordered={acc_ordered:.3f} acc_pooled={acc_pooled:.3f}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"gunpoint baseline: {error}", file=sys.stderr)
        raise SystemExit(1)
