"""CLI entry: python -m eml_sr recover --target exp"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from eml_sr.fit import fit


def _target_points(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "exp":
        x = np.array([0.1, 0.5, 1.0])
        return x, np.exp(x)
    if name == "poly":
        x = np.array([1.0, 2.0, 3.0, 4.0])
        return x, x * x + 1.0
    if name == "ln":
        x = np.array([1.0, 2.0, 3.0, np.exp(1.0)])
        return x, np.log(x)
    raise ValueError(f"unknown target: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eml_sr")
    sub = parser.add_subparsers(dest="cmd", required=True)

    recover = sub.add_parser("recover")
    recover.add_argument("--target", required=True, choices=["exp", "poly", "ln"])
    recover.add_argument("--max-depth", type=int, default=3)
    recover.add_argument("--method", default="enumerate", choices=["enumerate", "legacy-enumerate", "adam"])
    recover.add_argument("--domain", default="complex", choices=["real", "complex"])

    args = parser.parse_args(argv)
    if args.cmd != "recover":
        parser.error("only recover is implemented in v1")

    x, y = _target_points(args.target)
    result = fit(
        x,
        y,
        max_depth=args.max_depth,
        method=args.method,
        domain=args.domain,
    )
    print(f"mse={result.mse}")
    print(f"eml_nodes={result.eml_nodes}")
    print(f"rpn={result.rpn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
