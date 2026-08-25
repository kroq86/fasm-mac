"""High-level fit API for eml-sr."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from eml_sr import _core

Method = Literal["enumerate", "adam"]
Domain = Literal["real", "complex"]


@dataclass(frozen=True)
class FitResult:
    mse: float
    rpn: str
    eml_nodes: int
    _native: _core.FitResult

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = np.asarray(x, dtype=float)
        if xs.ndim == 0:
            return np.array(self._native.predict(float(xs)))
        flat = xs.reshape(-1)
        out = self._native.predict_many(flat.tolist())
        return np.asarray(out, dtype=float).reshape(xs.shape)

    def to_dot(self) -> str:
        return self._native.to_dot()


def fit(
    x,
    y,
    *,
    max_depth: int = 3,
    method: Method = "enumerate",
    domain: Domain = "complex",
    jobs: int = 1,
    profile: bool = False,
    epochs: int = 2000,
    lr: float = 0.05,
) -> FitResult:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    if x_arr.size == 0:
        raise ValueError("x and y must be non-empty")

    native = _core.fit(
        x_arr.reshape(-1).tolist(),
        y_arr.reshape(-1).tolist(),
        max_depth=max_depth,
        method=method,
        domain=domain,
        jobs=jobs,
        profile=profile,
        epochs=epochs,
        lr=lr,
    )
    return FitResult(
        mse=float(native.mse),
        rpn=str(native.rpn),
        eml_nodes=int(native.eml_nodes),
        _native=native,
    )


def eml(x: float, y: float) -> float:
    return float(_core.eml(x, y))
