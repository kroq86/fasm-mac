# eml-sr

Python library for **EML symbolic regression** — discrete search over binary trees built from

```text
eml(x, y) = exp(x) - ln(y)
grammar:   S → 1 | eml(S, S)
leaves:    1, x, f
```

Inspired by the EML operator and search ideas in
[arXiv:2603.21852](https://arxiv.org/abs/2603.21852) (Odrzywołek, 2026).
The canonical reference implementation is
[VA00/SymbolicRegressionPackage](https://github.com/VA00/SymbolicRegressionPackage)
by the paper author.

`eml-sr` v1 is a **pip-installable** `(x, y) → formula` engine (C++ core + pybind11).
It does **not** include Mathematica, PyTorch GPU training, CUDA recognizer, or
`VerifyBaseSet` from VA00.

## Install

```sh
pip install eml-sr
```

Development install from this monorepo:

```sh
pip install -e python
```

## Quick start

```python
import numpy as np
import eml_sr

x = np.array([0.1, 0.5, 1.0])
y = np.exp(x)
result = eml_sr.fit(x, y, max_depth=2)
print(result.mse, result.rpn)   # ~0, "x 1 eml"
print(result.predict(x))
```

## API (v1)

| Function | Description |
|----------|-------------|
| `eml_sr.fit(x, y, **opts)` | Search best EML tree; returns `FitResult` |
| `FitResult.predict(x)` | Evaluate fitted tree |
| `FitResult.to_dot()` | Graphviz DOT for the tree |
| `eml_sr.eml(x, y)` | Primitive `exp(x) - log(y)` |

Options: `max_depth` (default 3), `method` (`enumerate` \| `adam`), `domain` (`complex` \| `real`).

## vs VA00 SymbolicRegressionPackage

| Feature | VA00 (author reference) | eml-sr v1 |
|---------|-------------------------|-----------|
| EML tree search on `(x,y)` | yes (Rust/Mathematica) | yes (C++ core) |
| Adam / parametric leaves | yes (PyTorch) | minimal C++ Adam |
| Mathematica `RecognizeFunction` | yes | no |
| CUDA shortest-expression search | yes | no |
| pip install | no | yes |

Use **VA00** for paper-grade tooling and completeness checks.
Use **eml-sr** when you only need EML fit in Python without the full stack.

## Acknowledgments

- **Andrzej Odrzywołek** — EML operator, paper, and
  [SymbolicRegressionPackage](https://github.com/VA00/SymbolicRegressionPackage).
- This project implements a narrow search path compatible with the paper grammar;
  it is not affiliated with or endorsed by the VA00 repository.

## License

MIT
