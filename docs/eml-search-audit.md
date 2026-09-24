# EML search audit

Date: 2026-09-24. Host: Apple Silicon, macOS 15.7.2. Compiler: Apple clang 17.0.0. Flags: `-std=c++20 -O2`.

Classification: **B. Extractable, not package-ready.**

The hang and the memo lifetime are two separate facts. Do not write that fixing the memo made the hang go away.

> The hang is a use-after-free. After the node is copied before recursive `intern_*`, a search-lifetime memo stays correct and is much worse in memory and time.

## Bug fix

`instantiate_leaf_assignment` and `with_arena_f_values` held an `ExprNode&` into `ExprArena::nodes_` across `intern_*`. Those calls can reallocate the vector. The next child id was then read from freed storage. When that storage was reused, the walk became a cycle and did not return.

A sample of a stuck native process was entirely inside `with_arena_f_values`. A search-lifetime `ArenaEvalMemo` made that reuse stable enough for the undefined behavior to show up as a hang, including poly depth 1 (9 candidates). The earlier hypothesis “depth 4 hangs because the global memo grows” is not the cause of the non-termination.

Fix: copy the node before the recursive `intern_*`. After that copy, global-memo poly depth 1 returns in 0.23 ms.

## Performance policy

`ArenaEvalMemo` is keyed by expression hash and `x`. The hash includes probed `f` values. A memo that lives for the whole search keeps those probes after the candidate scope rolls the nodes back.

Default is `MemoLifetime::PerCandidate`. `SearchGlobal` remains only so the audit can compare the two.

Poly depth 4, after the copy, same machine, one run each:

```text
global:        966 ms    1,237,158 memo entries    109 MB
per-candidate: 464 ms        4,726 memo entries    4.8 MB
same 3870 candidates
same MSE 0.156508
same RPN  f f x eml eml x f eml eml
```

x86_64/Rosetta global depth 4: 1072 ms, 1,237,152 entries, 111 MB, same MSE and RPN.

## Growth

Per-candidate candidate counts on the poly points `(1,2), (2,5), (3,10), (4,17)`. No early stop. Ratios are about ×7, ×7.4, ×8.3, ×8.9.

```text
depth 1:        9
depth 2:       63
depth 3:      468
depth 4:    3,870
depth 5:   34,488
```

Depth 5 completes: arm64 4975 ms, max memo 7252, RSS 6.8 MB, MSE 0.0199931, RPN `f f x eml 1 1 eml eml eml f eml`. That is combinatorial growth, not a crash at depth 4.

Per-candidate medians, native arm64, repeated runs:

| case | median | candidates | max memo | MSE | RPN |
|---|---:|---:|---:|---:|---|
| exp depth 1 | 0.0048 ms | 2 | 9 | 0 | `x 1 eml` |
| poly depth 1 | 0.26 ms | 9 | 428 | 32.2537 | `f f eml` |
| poly depth 2 | 3.83 ms | 63 | 1537 | 18.3133 | `f f x eml eml` |
| ln depth 3 | 12.4 ms | 154 | 2908 | ~0 | `1 1 x eml 1 eml eml` |
| poly depth 3 | 40.7 ms | 468 | 3270 | 2.40127 | `1 f x eml eml f eml` |
| poly depth 4 | 441 ms | 3870 | 4726 | 0.156508 | `f f x eml eml x f eml eml` |
| poly depth 5 | 4975 ms | 34488 | 7252 | 0.0199931 | `f f x eml 1 1 eml eml eml f eml` |

Exp and ln stop at MSE `1e-9`. They keep the same candidate count, MSE, and RPN under both memo lifetimes.

## arm64 and x86_64

The pure C++ search does not need x86_64. `-arch x86_64` was a script default. `scripts/check_eml_sr.sh`, `scripts/check_eml_sr_fast.sh`, and `scripts/bench_eml_sr_compare.sh` now compile that search for the host. `fasm/core/eml.inc` and `fasm/apps/eml_core.asm` stay x86_64 and are not linked into this search.

On this Mac, these cases, native arm64 was faster than x86_64 under Rosetta:

```text
depth 4:  arm64 441–464 ms    Rosetta 771 ms
depth 5:  arm64 4975 ms       Rosetta 8598 ms
```

That is not a general architecture claim.

Selected MSE and RPN match. Eval counters do not. Poly depth 4 per-candidate `eml_calls`: arm64 1,893,662, x86_64 1,893,623. Same winner. Treat that as a floating-point prune threshold until someone needs a deterministic counter. It does not block a later private package cut.

## What is proven

```text
math/search idea          READY
correctness bug           FIXED
memo lifetime             FIXED
arm64 portability         PROVEN
core isolation            PROVEN
minimal C++ consumer      PROVEN

public C++ API            NOT READY
library target            MISSING
Python binding            NOT REVALIDATED
package name              COLLIDES
cross-arch exact counters NOT DETERMINISTIC
external benchmark        NOT DONE
```

The core is header-only C++ under `fasm/apps/eml_sr/`. It does not include PyTorch, the tensor engine, or `eml.inc`. `fasm/tests/eml_sr/minimal_consumer.cpp` compiles with only `-I fasm/apps/eml_sr` and prints the poly depth-4 result on arm64 and on x86_64.

A caller of `search_best` still includes three internal headers: `adam.hpp`, `arena_search.hpp`, `pool_search.hpp`. There is no installed facade, no library target, and no CMake export. The Python surface (`EMLRegressor`, `fit`, `FitResult`) was not rebuilt after these fixes. `python/pyproject.toml` still names the project `eml-sr`, which is already taken on PyPI. Do not publish under that name.

No comparison with the Odrzywołek paper, VA00, or PyPI `eml-sr` was run. The older in-repo note (legacy enumerator ~20.8 s, `ExprArena` ~1.2 s, one poly depth-4 case) is an internal historical comparison and was not re-run here.

## Expression search versus the EML task

The useful split is the enumerator versus this one task. Lean files are a source of a mathematical statement, not an input format. The question is whether enumeration, arena intern, dedup, memo, prune, and ranking can run with an external grammar and an external score.

They cannot, in this tree. `ExprKind` is `Leaf | Eml`. `gen_arena_shapes` emits only `intern_eml`. Leaf assignment is the fixed alphabet `1`, `x`, `f`. `eval_expr` always calls `eml`. The memo key is that expression hash plus one real `x`. The score in the inner loop is `mse_for_expr_ctx` on `DataPoint {x, y}`.

What is already independent of the formula `exp(a) - ln(b)`: intern, rollback, dedup by child ids, and enumeration of full binary trees of one operator. Swapping that one binary function at the eval site is a local change. Several operators, unary nodes, several variables, derivative nodes, or a residual that is not pointwise MSE need a different node and a different evaluator. That is a new search core that can copy this arena. It is not a facade over the current headers.

Classification B is unchanged. It applies to this EML fitter. A generic formula-search library is a separate design and is not extracted here.

## Next pass, when there is one

Not depth 6, not SIMD, not Metal, not assembly. The next pass is package readiness only:

```text
one public header
one library target
one stable SearchConfig / SearchResult API
CMake install/export
minimal external consumer through the installed library
Python binding rebuilt against that same public core
new package name
```

## Reproduce

```bash
scripts/bench_eml_search_audit.sh
```

Cases in that script time out at 8–20 seconds. Regression binary: `fasm/tests/eml_sr/eml_memo_regression.cpp`.

```bash
clang++ -std=c++20 -O2 -arch arm64 -I fasm/apps/eml_sr \
  fasm/tests/eml_sr/eml_search_audit.cpp -o /tmp/eml-search-audit
/tmp/eml-search-audit --memo per --case poly --depth 4 --repeats 3
/tmp/eml-search-audit --memo global --case poly --depth 4 --repeats 1
```
