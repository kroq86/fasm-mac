# Repository Working Rules

## Rule (summary)

- **`fasm/core/*.inc` is the growing stdlib** — reusable behavior lives here, not in apps.
- **`fasm/apps/*` are thin consumers** — products wire core helpers; no private mini-frameworks per app.
- **`fasm/examples/*` demonstrate core** — important algorithms and infrastructure must not hide in example-local code.
- **Core changes are high blast radius** — small diffs, caller audit before semantic edits, then broad checks (`scripts/check_*.sh`, macOS smoke, `git diff --check`); not done until every known consumer still builds and matches expected output.
- **Brew-worthy products beat toy examples** — finished binaries, real macOS/developer workflows, check scripts, and a release path (`Formula/`, `scripts/build-*-release.sh`).

## Complexity Checklist (humans and agents)

Before expanding scope or mixing layers, run this list — one concern at a time:

1. **Core or app?** Reusable behavior → `fasm/core/*.inc`; product wiring → `fasm/apps/*`; demos → `fasm/examples/*`. Do not blend syscall/Mach-O/ABI work with product logic in one place.
2. **Contract first.** Register args, return codes, struct offsets, and clobber rules in comments (`ccall64.inc` style). Prefer one extern + C ABI over ad hoc coupling.
3. **Additive and local.** Extend core with new helpers; audit callers before changing semantics. A change should touch few modules (`scripts/check_*.sh` locks the contract).
4. **Hide implementation, not obligations.** DSL/macros and stdlib helpers reduce cognitive load only when the boundary is explicit — not when complexity leaks across callers.
5. **Narrow the problem.** Stay within repo limits (x86_64 macOS, documented ABI). Reject extra platforms, backends, or “future-proof” layers unless explicitly requested.
6. **No parasitic complexity.** No app-local mini-frameworks, no drive-by refactors, no patterns that widen variability without a product need. Two failed fix attempts on the same issue → stop and hand off.

## Core-First Product Strategy

This repository is not just a collection of standalone FASM demos. Treat it as
a growing small standard library plus products built on top of it.

- Reusable behavior belongs in `fasm/core/*.inc`.
- Products in `fasm/apps/*` should be thin consumers of core helpers whenever a
  behavior is likely to be reused by another app, test, or future product.
- Examples in `fasm/examples/*` should demonstrate core abstractions rather
  than hide important algorithms or infrastructure locally.
- Scripts in `scripts/check_*.sh` are the regression gates for products and core
  areas.

When adding product functionality, first ask whether it should grow the core.
If the answer is yes, add or extend a focused core helper, then wire the product
to that helper. Avoid building a private mini-framework inside one app.

## Core Change Safety

Core changes can break every product, so make them small and verified.

- Keep core APIs assembly-friendly: explicit register contracts in comments,
  stable structs/offsets, simple return codes, caller-owned buffers where
  practical.
- Prefer additive helpers over changing existing helper semantics.
- If changing an existing core routine, audit all current callers before editing.
- After touching `fasm/core/*`, run the relevant product checks plus the broad
  smoke set:
  - `scripts/check_machodoctor.sh`
  - `scripts/check_logknife.sh`
  - `scripts/check_logvec.sh` (when touching `vec_f32.inc` or `logvec_payload.inc`)
  - `scripts/check_fscan.sh`
  - `scripts/check_leetcode_examples.sh`
  - affected macOS smoke tests under `fasm/tests/macos-smoke`
  - `git diff --check`
- Do not call a core refactor done until every known consumer still builds and
  runs with its expected output.

## Product Direction

Favor brew-worthy, ready-to-run macOS tools over toy examples. A good product in
this repo should:

- install as a finished binary, not require users to assemble it themselves;
- solve a real macOS/developer workflow problem;
- exercise and improve reusable core infrastructure;
- ship with a check script and release packaging path.

