#!/usr/bin/env python3
"""Dependency-free baseline for the open NIR-MFCO material-flow dataset.

The archive is intentionally kept outside the repository.  Labels are encoded
in filenames (``HDPE_0200`` means a 0.200 HDPE share).  Raw images use exact
false colours, so a sampled colour histogram is a compact numeric input for a
future tensor-runtime workload.
"""

import argparse
import hashlib
import math
import re
import struct
import sys
import zipfile

DATASET_MD5 = "6703e6d59b2972b0d0c16d21be6a811d"
NAME_RE = re.compile(
    r"^raw/(T1|T2a|T2b)/(SI|MO|H1|H2)/HDPE_(\d{4})_[^/]+_(\d+)\.bmp$"
)
GROUPS = [(series, presentation) for series in ("T1", "T2a", "T2b")
          for presentation in ("SI", "MO", "H1", "H2")]


def bmp_features(blob, stride=8):
    if blob[:2] != b"BM" or len(blob) < 54:
        raise ValueError("not a BMP image")
    offset = struct.unpack_from("<I", blob, 10)[0]
    header, width, height, planes, bits, compression = struct.unpack_from(
        "<IiiHHI", blob, 14
    )
    if header < 40 or width <= 0 or height == 0 or planes != 1 or bits != 24 or compression:
        raise ValueError("expected uncompressed 24-bit BMP")
    height = abs(height)
    row_bytes = (width * 3 + 3) & ~3
    counts = {"red": 0, "blue": 0, "orange": 0, "white": 0, "other": 0}
    for y in range(0, height, stride):
        row = offset + y * row_bytes
        for x in range(0, width, stride):
            b, g, r = blob[row + x * 3:row + x * 3 + 3]
            if (r, g, b) == (255, 0, 0):
                counts["red"] += 1
            elif (r, g, b) == (0, 0, 255):
                counts["blue"] += 1
            elif (r, g, b) == (255, 127, 0):
                counts["orange"] += 1
            elif (r, g, b) == (255, 255, 255):
                counts["white"] += 1
            else:
                counts["other"] += 1
    total = sum(counts.values())
    foreground = total - counts["white"]
    denom = max(1, foreground)
    return (counts["red"] / denom, counts["blue"] / denom,
            counts["orange"] / denom, foreground / total)


def bmp_sequence_features(blob, bands=3, stride=8):
    if blob[:2] != b"BM" or len(blob) < 54:
        raise ValueError("not a BMP image")
    offset = struct.unpack_from("<I", blob, 10)[0]
    header, width, height, planes, bits, compression = struct.unpack_from(
        "<IiiHHI", blob, 14
    )
    if header < 40 or width <= 0 or height == 0 or planes != 1 or bits != 24 or compression:
        raise ValueError("expected uncompressed 24-bit BMP")
    height = abs(height)
    row_bytes = (width * 3 + 3) & ~3
    counts = [[0, 0, 0, 0, 0] for _ in range(bands)]
    for y in range(0, height, stride):
        band = min(bands - 1, y * bands // height)
        row = offset + y * row_bytes
        for x in range(0, width, stride):
            b, g, r = blob[row + x * 3:row + x * 3 + 3]
            color = 0 if (r, g, b) == (255, 0, 0) else 1 if (r, g, b) == (0, 0, 255) else 2 if (r, g, b) == (255, 127, 0) else 3 if (r, g, b) == (255, 255, 255) else 4
            counts[band][color] += 1
    result = []
    for red, blue, orange, white, other in counts:
        total = red + blue + orange + white + other
        foreground = total - white
        denom = max(1, foreground)
        result.extend((red / denom, blue / denom, orange / denom,
                       foreground / max(1, total)))
    return result


def design(group, histogram):
    """Group-specific affine calibration, represented as one dense row."""
    group_index = GROUPS.index(group)
    red, blue, orange, occupancy = histogram
    local = (1.0, red, blue, orange, occupancy)
    row = [0.0] * (len(GROUPS) * len(local))
    start = group_index * len(local)
    row[start:start + len(local)] = local
    return row


def solve_ridge(rows, labels, ridge=1e-6):
    width = len(rows[0])
    a = [[0.0] * (width + 1) for _ in range(width)]
    for row, label in zip(rows, labels):
        for i, xi in enumerate(row):
            if xi:
                a[i][-1] += xi * label
                for j, xj in enumerate(row):
                    a[i][j] += xi * xj
    for i in range(width):
        a[i][i] += ridge
    for col in range(width):
        pivot = max(range(col, width), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        if abs(scale) < 1e-15:
            raise ValueError("singular regression system")
        for j in range(col, width + 1):
            a[col][j] /= scale
        for row in range(width):
            if row == col:
                continue
            scale = a[row][col]
            if scale:
                for j in range(col, width + 1):
                    a[row][j] -= scale * a[col][j]
    return [a[i][-1] for i in range(width)]


def predict(row, weights):
    return sum(x * w for x, w in zip(row, weights))


def evaluate(samples):
    # Repetition zero is held out in every physical series/presentation group.
    train = [sample for sample in samples if sample[2] != 0]
    test = [sample for sample in samples if sample[2] == 0]
    weights = solve_ridge([s[0] for s in train], [s[1] for s in train])
    errors = [predict(s[0], weights) - s[1] for s in test]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean = sum(s[1] for s in test) / len(test)
    ss_res = sum(error * error for error in errors)
    ss_tot = sum((s[1] - mean) ** 2 for s in test)
    return len(train), len(test), mae, rmse, 1.0 - ss_res / ss_tot


def load_archive(path, stride):
    samples = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            match = NAME_RE.match(name)
            if not match:
                continue
            series, presentation, share, repetition = match.groups()
            histogram = bmp_features(archive.read(name), stride)
            samples.append((design((series, presentation), histogram),
                            int(share) / 1000.0, int(repetition)))
    if len(samples) != 880:
        raise ValueError(f"expected 880 raw images, found {len(samples)}")
    return samples


def export_fixture(path, samples):
    """Write a small, stable float32 hand-off file for the x86 tensor spike."""
    with open(path, "wb") as output:
        output.write(struct.pack("<8sIII", b"NIRMFCO1", len(samples),
                                 len(samples[0][0]), 0))
        for row, label, repetition in samples:
            output.write(struct.pack("<I", repetition == 0))
            output.write(struct.pack(f"<{len(row)}f", *row))
            output.write(struct.pack("<f", label))


def export_sequence_fixture(path, archive_path, stride, bands=3):
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            match = NAME_RE.match(name)
            if not match:
                continue
            series, presentation, share, repetition = match.groups()
            features = bmp_sequence_features(archive.read(name), bands, stride)
            records.append((int(repetition) == 0, GROUPS.index((series, presentation)),
                            features, int(share) / 1000.0))
    if len(records) != 880:
        raise ValueError(f"expected 880 sequence records, found {len(records)}")
    with open(path, "wb") as output:
        output.write(struct.pack("<8sIIII", b"NIRSEQ1\0", len(records), bands, 4, 0))
        for test, group, features, label in records:
            output.write(struct.pack("<II", test, group))
            output.write(struct.pack(f"<{len(features)}f", *features))
            output.write(struct.pack("<f", label))


def self_test():
    rows, labels = [], []
    for group in GROUPS:
        for repetition, red in enumerate((0.1, 0.2, 0.3)):
            rows.append((design(group, (red, 1.0 - red, 0.0, 0.5)),
                         2.0 * red + 0.01, repetition))
            labels.append(2.0 * red + 0.01)
    train, test, mae, _, _ = evaluate(rows)
    if train != 24 or test != 12 or mae > 1e-4:
        raise AssertionError(f"baseline self-test failed: mae={mae}")
    print("nir-mfco baseline self-test passed: groups=12 split=repetition-0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", nargs="?")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--verify-md5", action="store_true")
    parser.add_argument("--export")
    parser.add_argument("--export-sequence")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.archive:
        parser.error("archive is required unless --self-test is used")
    if args.stride < 1:
        parser.error("--stride must be positive")
    if args.verify_md5:
        digest = hashlib.md5(open(args.archive, "rb").read()).hexdigest()
        if digest != DATASET_MD5:
            raise ValueError(f"dataset checksum mismatch: {digest}")
    samples = load_archive(args.archive, args.stride)
    if args.export:
        export_fixture(args.export, samples)
    if args.export_sequence:
        export_sequence_fixture(args.export_sequence, args.archive, args.stride)
    train, test, mae, rmse, r2 = evaluate(samples)
    print(f"nir-mfco baseline samples={len(samples)} train={train} test={test} "
          f"features={len(samples[0][0])} split=repetition-0 stride={args.stride} "
          f"mae_pp={mae * 100:.3f} rmse_pp={rmse * 100:.3f} r2={r2:.5f}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"nir-mfco baseline: {error}", file=sys.stderr)
        raise SystemExit(1)
