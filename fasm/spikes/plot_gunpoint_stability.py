#!/usr/bin/env python3
"""Render the GunPoint numerical-stability CSV as a paper-ready SVG."""

import csv
import math
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} stability.csv output.svg", file=sys.stderr)
        return 2
    rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8")))
    if not rows:
        return 2
    points = [(int(r["update"]), max(float(r["state_max_abs"]), 1e-12)) for r in rows]
    width, height = 960, 540
    left, right, top, bottom = 92, 28, 42, 72
    pw, ph = width - left - right, height - top - bottom
    xmax = max(x for x, _ in points)
    ymin, ymax = -12.0, math.ceil(math.log10(max(y for _, y in points)))

    def sx(x: float) -> float:
        return left + pw * x / xmax

    def sy(y: float) -> float:
        return top + ph * (ymax - math.log10(max(y, 1e-12))) / (ymax - ymin)

    path = " ".join(("M" if i == 0 else "L") + f"{sx(x):.2f},{sy(y):.2f}" for i, (x, y) in enumerate(points))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#18212f}.axis{stroke:#657083;stroke-width:1}.grid{stroke:#dce1e8;stroke-width:1}.trace{fill:none;stroke:#1769aa;stroke-width:2}.mark{stroke:#c43d3d;stroke-width:1.5;stroke-dasharray:5 4}.label{font-size:13px}.title{font-size:19px;font-weight:600}</style>',
        '<text class="title" x="92" y="26">GunPoint: long-horizon float32 parameter divergence</text>',
    ]
    for exponent in range(int(ymin), int(ymax) + 1, 2):
        y = sy(10.0**exponent)
        out += [f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>', f'<text class="label" text-anchor="end" x="{left-10}" y="{y+4:.2f}">10^{exponent}</text>']
    for x in range(0, xmax + 1, 25000):
        px = sx(x)
        out += [f'<line class="grid" x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{height-bottom}"/>', f'<text class="label" text-anchor="middle" x="{px:.2f}" y="{height-bottom+24}">{x:,}</text>']
    out += [
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
        f'<path class="trace" d="{path}"/>',
    ]
    for update, label, anchor in ((2, "first persistent-state divergence (1 ULP)", "start"), (10874, "first material logit divergence", "start")):
        x = sx(update)
        text_x = x + 7
        out += [f'<line class="mark" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>', f'<text class="label" text-anchor="{anchor}" x="{text_x:.2f}" y="{top+20 if update == 2 else top+42}">{label} · update {update:,}</text>']
    out += [
        f'<text class="label" text-anchor="middle" x="{left+pw/2:.2f}" y="{height-18}">Training update</text>',
        f'<text class="label" text-anchor="middle" transform="translate(22 {top+ph/2:.2f}) rotate(-90)">Maximum absolute parameter-state delta (log scale)</text>',
        '</svg>',
    ]
    Path(sys.argv[2]).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"GunPoint stability plot written: {sys.argv[2]} points={len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
