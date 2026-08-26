#!/usr/bin/env python3
import math, struct, sys
from nir_mfco_baseline import solve_ridge

def main(path):
    with open(path, "rb") as f:
        magic, rows, tokens, features, _ = struct.unpack("<8sIIII", f.read(24))
        if magic[:7] != b"NIRSEQ1" or (rows, tokens, features) != (880, 3, 4):
            raise ValueError("bad sequence fixture")
        samples = []
        for _ in range(rows):
            test, group = struct.unpack("<II", f.read(8))
            values = struct.unpack("<12f", f.read(48))
            label, = struct.unpack("<f", f.read(4))
            pooled = [sum(values[t * 4 + j] for t in range(3)) / 3 for j in range(4)]
            row = [0.0] * 60
            row[group * 5:group * 5 + 5] = [1.0] + pooled
            samples.append((test, row, label))
    train = [s for s in samples if not s[0]]
    test = [s for s in samples if s[0]]
    weights = solve_ridge([s[1] for s in train], [s[2] for s in train])
    errors = [sum(x*w for x,w in zip(s[1],weights))-s[2] for s in test]
    mae = sum(map(abs, errors))/len(errors)
    print(f"nir-mfco pooled-band linear: train={len(train)} test={len(test)} mae_pp={mae*100:.3f} order_invariant=yes")

if __name__ == "__main__":
    main(sys.argv[1])
