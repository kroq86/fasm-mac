# eml_sr — EML symbolic regression (standalone)

Research tool for **formula discovery** using the EML operator from
[arXiv:2603.21852](https://arxiv.org/abs/2603.21852) (Odrzywołek, 2026).

```text
eml(x, y) = exp(x) - ln(y)     on std::complex<double>
grammar:   S → 1 | eml(S, S)
leaves:    1, x, f (scalar parameter fit per tree)
```

This is **not** part of the logbus → logvec → ragbox product stack.

| | ragbox | eml_sr |
|---|--------|--------|
| Binary | `ragbox` | `eml_sr` |
| Brew | yes | **no** |
| FASM | logvec_core.o | **no** (libm only) |
| Checks | `check_ragbox*` | **`check_eml_sr.sh` only** |

## What it does

```text
(x_i, y_i)
  → enumerate EML binary trees (eml node count ≤ K)
  → eval on complex doubles
  → MSE → best tree
  → stdout: RPN + MSE
  → optional Graphviz DOT (--dot)
```

v1 uses **discrete search** over tree shape and leaf assignment `{1, x, f}` with
coordinate descent on `f` parameters. Search also supports **Adam** training on a
fixed left-heavy tree shape (`--method adam`).

## Search methods

| Method | Flag | Notes |
|--------|------|-------|
| Enumerate | `--method enumerate` (default) | Bottom-up pool + full shape enumeration, memoized eval |
| Adam | `--method adam` | Finite-diff Adam on parametric leaf logits, snap to discrete tree |

## Performance flags

```text
--profile          print SearchStats after search (forms, candidates, eml_calls, pruned, cache_hit)
--domain real      real-axis exp/log eval (faster for positive x)
--domain complex   default; full complex intermediates
--jobs N           OpenMP parallel assignment loop (requires -fopenmp build)
--epochs N         Adam training epochs (default 2000)
--max-depth N      default 3; depth 4 is slow (~90s pre-opt, less after memo/dedup)
```

Baseline before optimization (legacy `EML_SR_LEGACY=1`):

```text
recover exp     instant
recover poly --max-depth 4   ~90s
```

With `--profile`:

```text
forms=... candidates=... eml_calls=... pruned=... cache_hit=... best_mse=...
```

Legacy discrete search remains available via `EML_SR_LEGACY=1`.

## Build

```sh
clang++ -std=c++20 -O2 -arch x86_64 fasm/apps/eml_sr/eml_sr.cpp -o eml_sr
```

OpenMP build (optional, macOS needs `brew install libomp`):

```sh
clang++ -std=c++20 -O2 -arch x86_64 -Xpreprocessor -fopenmp \
  -I/opt/homebrew/opt/libomp/include \
  fasm/apps/eml_sr/eml_sr.cpp -o eml_sr \
  -L/opt/homebrew/opt/libomp/lib -lomp
```

### FASM core leaf (Level 1–2)

Separate from the C++ search tool: a replaceable **real f64** primitive in
[`fasm/core/eml.inc`](../fasm/core/eml.inc), exported by
[`fasm/apps/eml_core.asm`](../fasm/apps/eml_core.asm):

```sh
fasm --emit=macho-obj fasm/apps/eml_core.asm eml_core.o
clang++ -std=c++20 -O2 -arch x86_64 fasm/tests/eml_sr/eml_core_smoke.cpp eml_core.o -lm -o eml_core_smoke
```

C ABI: `double lb_eml_f64(double x, double y)` — `xmm0`/`xmm1` in, result in
`xmm0`. Matches the paper identity `eml(x,1) = exp(x)` on the real axis.
Full paper semantics (complex intermediates, ln on negative args) stay in the
C++ `eml_sr` tool for now.

Header: [`fasm/apps/eml_core.hpp`](../fasm/apps/eml_core.hpp).

## CLI

```text
eml_sr verify
eml_sr recover --target exp|poly [--max-depth N] [--dot PATH]
eml_sr fit-bench [--max-depth N] [--dot PATH] [--points FILE]
```

Examples:

```sh
eml_sr verify
eml_sr show --preset ln --dot-eval 2 --dot /tmp/tree_ln.dot   # full 3-node paper tree
eml_sr recover --target exp --dot /tmp/tree.dot                # minimal: 1 eml node
eml_sr recover --target poly --max-depth 4 --dot /tmp/tree_poly.dot  # ~90s, 4 eml nodes
eml_sr fit-bench --points fasm/tests/eml_sr/exp_points.txt
```

`recover --target exp` finds **minimal** tree `eml(x,1)` — only **1 eml node** (2 leaves).
For a **full computation tree** from the paper use `show --preset ln` (3 eml nodes) or
`recover --target poly` (4 eml nodes, slow). Add `--dot-eval 2.0` to annotate each node
with the computed value at `x=2`.

`fit-bench` reads whitespace-separated `x y` rows from `--points FILE` and/or stdin.
It also accepts optional **bench_perf-style** latency lines on stdin
(`count layer median_ms`, e.g. `10000 search 4.2`) and echoes them as
`latency_count=… latency_layer=… latency_ms=…`. No ragbox dependency.

```sh
# optional: pipe bench_perf search rows while fitting on poly points
scripts/bench_perf.sh 2>/dev/null | rg '^[0-9]+ search ' \
  | eml_sr fit-bench --points fasm/tests/eml_sr/poly_points.txt
```

## Graphviz

```sh
eml_sr recover --target exp --dot tree.dot
dot -Gdpi=150 -Tpng tree.dot -o tree.png    # if graphviz installed
open tree.png
```

DOT nodes: `eml` ovals (`exp-left - ln-right`), leaves `1` / `x` / `f` boxes.
Edges are labeled `left/exp` and `right/ln` (non-commutative chirality).

## Python / PyPI

Install the pip package from [`python/`](../python/):

```sh
pip install -e python
python -c "import eml_sr, numpy as np; x=np.array([0.1,0.5,1.0]); print(eml_sr.fit(x,np.exp(x),max_depth=2).rpn)"
```

See [`python/README.md`](../python/README.md) for API details.

**Scope:** v1 exposes `eml_sr.fit(x, y)` over the C++ engine. Canonical EML research tooling
(Mathematica, PyTorch GPU training, CUDA search, `VerifyBaseSet`) lives in
[VA00/SymbolicRegressionPackage](https://github.com/VA00/SymbolicRegressionPackage)
by Andrzej Odrzywołek (paper author). `eml-sr` is inspired by that work, not a replacement.

```sh
scripts/check_eml_sr_python.sh   # pytest smoke (exp + eml primitive)
pytest python/tests -m slow      # optional depth-4 poly (~30s)
```

## Check

```sh
scripts/check_eml_sr_fast.sh   # ~5s: verify, exp, fit-bench (no poly)
scripts/check_eml_sr.sh        # full (~2 min): includes recover poly depth 4
CHECK_EML_SR_SKIP_POLY=1 scripts/check_eml_sr.sh   # skip slow poly step
```

**Do not** use `fit-bench --max-depth 4` on `poly_points.txt` for a quick smoke —
`recover poly` and that fit path each run ~90s discrete search at depth 4.

Quick manual smoke:

```sh
clang++ -std=c++20 -O2 -arch x86_64 fasm/apps/eml_sr/eml_sr.cpp -o eml_sr
./eml_sr verify
./eml_sr recover --target exp --dot /tmp/tree.dot
printf '1000 search 1.5\n' | ./eml_sr fit-bench --max-depth 1 --points fasm/tests/eml_sr/exp_points.txt
```

Builds `eml_sr`, runs checks above; full script also builds FASM `eml_core.o`.
**Not** wired into `check_ragbox.sh` or release checks.

## Bounded context

```mermaid
flowchart LR
  Data["(x, y) points"]
  Search["discrete EML tree search"]
  Eval["eml eval complex128"]
  Out["RPN + MSE + DOT"]
  Data --> Search --> Eval --> Out
```

## Layout

```text
fasm/apps/eml_sr/
  eml.hpp           # eml(x,y), near()
  tree.hpp          # Tree, to_rpn(), to_dot()
  eval.hpp          # cached eval, --domain real|complex, numeric prune
  stats.hpp         # SearchStats counters
  pool_search.hpp   # bottom-up pool + gen_shapes enumeration
  search.hpp        # search_best API, MSE, optimize_f
  adam.hpp          # parametric tree + Adam snap
  eml_sr.cpp        # CLI
fasm/core/eml.inc
fasm/apps/eml_core.asm
fasm/apps/eml_core.hpp
fasm/tests/eml_sr/
  expected_verify.txt
  exp_points.txt
  poly_points.txt
  eml_core_smoke.cpp
scripts/check_eml_sr.sh
```

## Reference

- Paper: [All elementary functions from a single operator](https://arxiv.org/abs/2603.21852)
- EML toolkit: [VA00/SymbolicRegressionPackage](https://github.com/VA00/SymbolicRegressionPackage)
