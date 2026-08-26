# fasm-mac

**ragbox is the flagship tool here:** local-first codebase memory for AI
agents. Build a searchable semantic snapshot of your repo, query it from the
terminal, and keep it local. Ollama-compatible, single binary, no vector DB
server.

This repository is the engineering monorepo behind ragbox and a small set of
macOS developer tools built on a reusable FASM runtime.

| Tool | Promise |
|------|---------|
| [`ragbox`](docs/ragbox.md) | Local codebase memory for Codex, Claude, Gemini, and other AI agents |
| `machodoctor` | Explain why a macOS binary does not run |
| `fasm-mac` | Assembly/runtime foundation for x86_64 FASM tools on macOS |

Product page: <https://kroq86.github.io/fasm-mac/>

## ragbox quick start

Use ragbox when `ripgrep` is too literal, a vector DB is too much machinery,
and you want a local semantic index your AI agent can query.

```sh
brew tap kroq86/fasm-mac https://github.com/kroq86/fasm-mac
brew install ragbox
brew install ollama
ollama pull nomic-embed-text

arch -x86_64 ragbox build --root . --out memory.lv
arch -x86_64 ragbox search --index memory.lv --query "where is auth handled?" --json
```

On Apple Silicon, ragbox runs through Rosetta (`arch -x86_64`). The index stays
in local files: `memory.lv`, `memory.lv.manifest.json`, optional refresh state,
and optional delta sidecar.

Why not the obvious alternatives?

| Alternative | ragbox difference |
|-------------|-------------------|
| `ripgrep` | semantic search, not lexical search |
| vector DB server | copyable file snapshot, not a running service |
| RAG platform | local CLI for repo memory, not a web platform |

More: [`docs/ragbox.md`](docs/ragbox.md). Release check:
`scripts/check_ragbox_release.sh`.

## fasm-mac foundation

fasm-mac is also an experimental macOS bridge for **flat assembler classic
1.73.35**. The goal is practical CLI compatibility for small x86_64 fasm
programs on macOS:

```sh
fasm file.asm
./file
```

On Apple Silicon this runs through Rosetta. This is not a native arm64 rewrite
of fasm classic.

## What This Is

fasm classic does not contain a Mach-O formatter. This project keeps the
upstream fasm compiler core and adds a macOS pipeline around it:

1. fasm emits ELF64 executable or ELF64 relocatable object output.
2. `fasm/tools/elf64_to_macho64.py` converts the supported ELF layout into
   Mach-O64.
3. macOS runs or links the resulting x86_64 Mach-O file.

That means this repository gives you a usable macOS command, not a new
upstream `format Mach-O` directive inside fasm.

## Install

### Homebrew (recommended)

```sh
brew tap kroq86/fasm-mac https://github.com/kroq86/fasm-mac
brew install fasm-mac
```

Then use `fasm` from anywhere:

```sh
fasm hello.asm
arch -x86_64 ./hello
```

On Apple Silicon, output binaries are x86_64 Mach-O and run through Rosetta.

Upgrade or remove:

```sh
brew upgrade fasm-mac
brew uninstall fasm-mac
```

### Manual install

```sh
./install.sh
```

The installer creates:

```text
~/.local/bin/fasm -> <repo>/bin/fasm
```

Make sure `~/.local/bin` is in your `PATH`:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

## CLI

Build a runnable Mach-O executable, using the source basename as output:

```sh
fasm fasm/basic/fib.asm
./fasm/basic/fib
```

Build and run:

```sh
fasm run fasm/basic/fib.asm
```

Explicit modes:

```sh
fasm --emit=macho file.asm [output]       # Mach-O executable
fasm --emit=elf file.asm [output]         # original ELF output
fasm --emit=macho-obj file.asm object.o   # Mach-O object for clang/dylib
```

Directly call the bundled host fasm:

```sh
fasm host <args...>
```

## macOS Includes

Runnable examples should include the platform layer instead of hardcoding Linux
syscall numbers:

```asm
format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable
entry start

start:
    write_file STDOUT, msg, msg_len
    exit 0

segment readable writeable
msg db "hello", 10
msg_len = $ - msg
```

The shim passes:

- `TARGET_OS=macos` for Mach-O output
- `TARGET_OS=linux` for `--emit=elf`

`platform.inc` currently provides `write_file`, `read_file`, `open_file`,
`close_file`, `exit`, syscall helper macros, and the Darwin constants needed by
the reusable file, socket, stat, and directory-walk helpers.

## Standard Library Includes

Platform-agnostic helpers for `format ELF64 executable 3`. All assemble
without data-segment declarations and follow the System V AMD64 ABI.

| File | What it provides |
|------|-----------------|
| [`fasm/core/for.inc`](fasm/core/for.inc) | `for reg, lo, hi` / `endfor reg` — nestable register-based counted loop |
| [`fasm/core/control.inc`](fasm/core/control.inc) | `_if/_else/_endif`, `_while/_endw`, `_repeat/_until` structured control flow |
| [`fasm/core/mem.inc`](fasm/core/mem.inc) | `memcpy`, `memset`, `memcmp`, `memmove`, `memxor` |
| [`fasm/core/base64.inc`](fasm/core/base64.inc) | `base64_encode(rdi,rsi,rdx)→rax`, `base64_decode(rdi,rsi,rdx)→rax` |
| [`fasm/core/math_fp.inc`](fasm/core/math_fp.inc) | `fp_isnan`, `fp_isinfinite`, `fp_isfinite`, `fp_floor`, `fp_ceil`, `fp_fmod`, `fp_frexp` |
| [`fasm/core/rational.inc`](fasm/core/rational.inc), [`fasm/core/polynomial.inc`](fasm/core/polynomial.inc) | exact `Rational` arithmetic and caller-owned polynomial operations |
| [`fasm/core/eml.inc`](fasm/core/eml.inc) | `lb_eml_f64` — EML operator leaf `exp(x)-log(y)` ([arXiv:2603.21852](https://arxiv.org/abs/2603.21852)); export via [`eml_core.asm`](fasm/apps/eml_core.asm) |

Examples:

```sh
fasm fasm/examples/elfexe/forloop64.asm /tmp/forloop64
fasm fasm/examples/elfexe/control64.asm /tmp/control64
```

## Structures and nested dynamic lists

fasm `struc` defines **memory layout** at assemble time (field offsets and
sizes). Runtime growable vectors live in [`fasm/core/dynvec.inc`](fasm/core/dynvec.inc)
with a central [`fasm/core/heap.inc`](fasm/core/heap.inc) allocator and
mark-and-sweep `gc_collect`.

Runnable demo (prints `[[1, 2], [3]]`):

```sh
fasm fasm/examples/nested_list_demo.asm
arch -x86_64 ./fasm/examples/nested_list_demo
```

Smoke test:

```sh
fasm fasm/tests/macos-smoke/nested_list.asm
arch -x86_64 ./fasm/tests/macos-smoke/nested_list
```

LeetCode-style examples covering arrays, hash maps, linked lists, trees,
graphs, stacks, and dynamic programming. Several examples are intentionally
thin wrappers around reusable helpers in `fasm/core`.

```sh
scripts/check_leetcode_examples.sh
```

### MLP primitives

`fasm/core/mlp_f32.inc` provides composable C-ABI primitives for an affine
dense layer (`Wx+b`) and an independent ReLU activation. The two-layer
`fasm/examples/mlp_forward.asm` example keeps model weights and wiring outside
core, leaving the operations suitable for a future autograd tape.

```sh
scripts/check_mlp_f32.sh
```

`fasm/core/autograd_f32.inc` is an experimental caller-owned scalar computation
tape with reverse-mode gradients for leaf values, addition, multiplication,
and ReLU. It is a correctness/reference spike, not the tensor training model;
the constraints and pre-XOR gates are recorded in `docs/autograd-spikes.md`.

```sh
scripts/check_autograd_f32.sh
```

### tensorctl — graph-planned tensor training engine

Everything from here through the GunPoint spike below is one connected
system, not a pile of unrelated experiments: a tensor representation with
views/strides, a semantic graph, a planner, an x86_64 assembly executor that
runs real kernels, finite-difference-checked forward/backward, an optimizer
loop, and a memory planner that makes actual cost-based decisions instead of
hardcoded policy. `tensorctl` is the v0.1 entry point into that system —
train a real model through it without reading any spike file:

```sh
scripts/build-tensorctl.sh
arch -x86_64 fasm/build/out/tensorctl mlp
arch -x86_64 fasm/build/out/tensorctl transformer
arch -x86_64 fasm/build/out/tensorctl transformer --plan                       # inspect the plan, no training run
arch -x86_64 fasm/build/out/tensorctl transformer --plan --memory-budget=8192  # watch the save/rematerialize decision flip
```

`mlp` trains the same 2→4→1 XOR network `fasm/examples/xor_tensor_train.asm`
already trains, but through the transformer's executor instead — proof this
engine isn't Transformer-specific (`tensor_mlp_train_check.c` below).
`transformer` trains the real `T=3,M=4,H=2,D=2,F=6` encoder block on a
synthetic target through the same backward schedule. `--plan` skips training
and prints the execution plan instead: action counts, a memory report
genuinely computed at run time (the same real 31-buffer arena layout from
`tensor_liveness_cost_planner_check.c`, with the full buffer table for
whichever schedule — save or rematerialize — the current `--memory-budget`
actually picks), and a second real alternative the planner compares:
standard merge+`CONTIGUOUS`+matmul versus the layout-aware head-wise
projection. This second decision is grounded in measured wall-clock, not
the byte-traffic count that looks better on paper (0 bytes copied vs 48) —
min/median/max over 31 trials, anti-dead-code-elimination-protected,
against real model state from an actual forward pass — and it genuinely
drives execution: `case7`+`case8` of the real 21-action backward schedule
are spliced out for one layout-aware step (21 actions become 20) when the
measurement favors it, still running through the exact same
`tensor_transformer_steps_execute` executor, not a bypass. Both paths are
verified equivalent every run — same forward output, same all five
gradients, and `b->merge` is poisoned before a layout-aware forward pass and
checked untouched after, proving the copy really doesn't happen. When the
gap between the two measured medians is smaller than their combined
min/max spread, the decision is reported `uncertain` and falls back to the
longer-established standard path instead of trusting a coin flip.

That 31-trial benchmark is real but doesn't need re-running on every
invocation: it's backed by a small autotuning cache (an XDG-respecting user
cache directory, never the repo), keyed on shape + host architecture +
this file's own revision marker + compiler identity + build flags — not a
whole-repo hash, so an unrelated edit elsewhere can't invalidate it, and
not auto-expired by age, since a three-day-old measurement isn't wrong just
for being old. Two different build configs (say `-O2` vs `-O0`) get
separate cache entries rather than one clobbering the other. The cache is
advisory only — written atomically (temp file + rename), and any read
trouble (missing, truncated, corrupt, wrong key) just falls back to a fresh
measurement, never a crash. `--reprofile` forces a fresh one anyway;
`--no-profile-cache` skips the cache file entirely, both directions.
Forward and the optimizer step are currently direct (not yet scheduled
through the executor for either model); the backward path is what's real:

```sh
scripts/check_tensorctl.sh
```

The rest of this section is the experiment history that got this far —
each spike below answers one specific question (is the planner general, does
rematerialization move the memory peak, does layout-aware kernel selection
help, is manual SIMD worth it) rather than adding a feature for its own
sake, several with real negative results left in rather than edited out.

The experimental scalar tensor ABI lives in `fasm/core/tensor_runtime_f32.inc`
with matmul, bias-add, ReLU, MSE, indexed-tape execution, lifetime validation,
and SGD implementations in focused `tensor_*_f32.inc` modules. Kernel calling
conventions are under test; planner, action, and ownership ABIs are not frozen.
A complete two-layer XOR training example consumes those helpers:

```sh
scripts/check_xor_tensor.sh
```

Build a persistent experimental training executable (x86_64 macOS; Rosetta is
used when running it on Apple Silicon):

```sh
scripts/build-xor-tensor.sh
arch -x86_64 fasm/build/out/xor-tensor-train
```

A larger experimental classifier learns all ten decimal digits from their
seven-segment display patterns with a `7 → 16 → 10` network:

```sh
scripts/build-digits-tensor.sh
arch -x86_64 fasm/build/out/digits-tensor-train
```

The NIR-MFCO spike exercises the same compiled tensor runtime on 880 open
industrial material-flow images. A dependency-free extractor converts exact
NIR false colours into 60 numeric features, holds out repetition zero from
each physical setup, and compares a linear calibration baseline with a
`60 → 16 → 1` fused tensor model. The 275 MB dataset remains external:

```sh
NIR_MFCO_ARCHIVE=/path/to/NIR-MFCO-dataset-v1.0.0.zip \
  scripts/check_nir_mfco_spike.sh
```

Without the archive, the script runs only its small deterministic contract
test. The external archive is identified by Zenodo record `7638775` and MD5
`6703e6d59b2972b0d0c16d21be6a811d`.

A single-head scaled dot-product attention spike composes the existing matmul
kernels with a materialized transpose and stable row-wise softmax. It checks
all 7/8/9 combinations for token, feature, and projection dimensions, compares
forward output with an independent scalar implementation, and checks backward
gradients by finite differences:

```sh
scripts/check_tensor_attention_spike.sh
```

This deliberately remains a 2D spike: batch and multi-head layouts require a
separate stride/rank ABI decision rather than silently widening `TensorView`.

That ABI decision is explored separately by a sidecar-layout spike. It keeps
the legacy 32-byte `TensorView`, stores rank/dimensions/strides and the lifetime
token in arena metadata, and passes a 16-byte `TensorRef` to strided kernels.
The check exercises batched matmul and O(1) transposed views across 7/8/9
dimensions without declaring the candidate ABI stable:

```sh
scripts/check_tensor_layout_stride_spike.sh
```

The follow-up multi-head attention kernel consumes rank-4
`[batch, heads, tokens, head_dim]` views directly. Its transposed K operand
shares storage through strides, and forward/backward cover every 7/8/9
combination for token and head dimensions:

```sh
scripts/check_tensor_multihead_attention_spike.sh
```

QKV projection, head splitting/merging, residual connections, and LayerNorm
remain separate transformer-block spikes.

The QKV layout check proves that one packed `[B*T, 3*H*D]` projection and its
three split-head views need no copies. Merging `[B,H,T,D]` back to
`[B*T,H*D]` is not a legal zero-copy reshape for that stride order, so the
planner must emit an explicit contiguous/permutation action. Residual-add and
LayerNorm are checked separately with saved row means/inverse standard
deviations and finite-difference backward validation:

```sh
scripts/check_tensor_qkv_layout_spike.sh
scripts/check_tensor_layernorm_spike.sh
```

The first integrated encoder-block forward then connects packed QKV, strided
multi-head attention, contiguous head merging, output projection, two residual
LayerNorm stages, and a `8 → 16 → 8` feed-forward network. Its 11-step plan is
checked at 7/8/9 token lengths; end-to-end block backward is still explicitly
pending:

```sh
scripts/check_tensor_transformer_block_spike.sh
```

End-to-end encoder backward is checked on a smaller numerical-oracle shape.
The gradient chain crosses both LayerNorms, the feed-forward network, output
projection, softmax attention, and packed QKV projection. Input and all five
weight groups are compared with finite differences of the complete block loss:

```sh
scripts/check_tensor_transformer_backward_spike.sh
```

The transformer planner spike lowers a 19-node semantic encoder graph into
explicit view, contiguous, attention, matmul, residual, LayerNorm, rematerialize,
and zero-gradient actions. It records saved-tensor and conservative scratch
contracts without declaring the new action or layout ABI stable:

```sh
scripts/check_tensor_transformer_planner_spike.sh
```

An x86_64 assembly executor consumes the emitted experimental 32-byte direct-
call steps. Its contract check runs the complete 14-action forward and
21-action backward schedules, including view, contiguous, rematerialize, and
zero-gradient actions, and verifies error-stop and stale-context behavior.
The callbacks are contract probes; transformer kernel contexts are not wired
into this executor yet:

```sh
scripts/check_tensor_transformer_executor_spike.sh
```

Because Apple Silicon executes the repository's x86_64 binaries through
Rosetta and cannot use their AVX2/FMA path, a native arm64 NEON feasibility
spike compares four-lane matmul and residual-ReLU kernels with their scalar
references and 1/7/8/9 tails. It reports both an optimized compiler baseline
and a diagnostic build with auto-vectorization disabled. This is evidence for
a possible arm64 backend, not an expansion of the stable platform contract:

```sh
scripts/check_tensor_neon_spike.sh
```

A compile/runtime dispatch-table spike exposes the same matmul, residual,
bias, ReLU, zero-gradient, and SGD families through scalar, NEON, and AVX2
slots. On Apple Silicon the native arm64 binary selects NEON, while the x86_64
binary under Rosetta selects scalar because AVX2/FMA are unavailable. Forced
scalar remains the correctness oracle and all backends retain scalar tails:

```sh
scripts/check_tensor_kernel_dispatch_spike.sh
```

Real encoder forward contexts are wired to the 14 emitted actions in a
follow-up executor integration check. The native arm64 path selects NEON; the
x86_64 assembly executor under Rosetta selects scalar. Both execute QKV,
attention, contiguous head merging, projections, two LayerNorms, and the FFN,
and match the forced-scalar block output. Backward action contexts remain
pending:

```sh
scripts/check_tensor_transformer_executor_kernels_spike.sh
```

That check's own `matmul` sub-timing at the block's real `T=3` shape was read
as "NEON loses to scalar on small shapes," which shaped an earlier size-aware-
dispatch plan. A dedicated size-sweep profiling spike falsifies that reading.
It runs the same `scalar_*`/`native_*` kernel pair from the dispatch spike
across many sizes, compiled two ways: plain `-O3` (`scalar_*` may get silently
auto-vectorized by clang, so this really compares hand NEON against whatever
the compiler gives you for free) and `-fno-vectorize -fno-slp-vectorize`
(`scalar_*` is genuinely scalar):

```sh
scripts/check_tensor_kernel_dispatch_profile.sh
```

At `m=3, n=12` — the exact matmul shape inside the real block — hand NEON
beats true scalar by 2.8-3.1x and keeps winning down to the smallest size
tested (crossover at n=4 for every family: matmul, residual, relu, zero,
bias, sgd). Against the plain `-O3` build, hand NEON *loses* at that same
shape (0.9x), matching the block benchmark almost exactly. So the earlier
result was real, but its explanation was wrong: it isn't that vectorization
doesn't pay off at small transformer shapes, it's that clang's own
auto-vectorizer already beats these particular hand-written NEON intrinsics
(a single unrolled 4-lane FMA per step, no accumulator or scheduling tricks)
on most shapes tested, small and large alike — re-running the block
benchmark itself even flipped sign between two back-to-back runs (0.82x,
then 1.44x), which is consistent with a close, noisy comparison rather than
a robust small-shape effect. The corrected implication is not "add a
scalar/NEON size threshold" but "the current hand-NEON kernels need real
optimization (unrolling, multiple accumulators) before they're worth
dispatching to at all, or the compiler's own vectorization should be
trusted instead of hand intrinsics here."

The matching backward integration emits 21 real actions: nine scoped gradient
zeroes, eleven reverse kernels, and one attention rematerialization. The
x86_64 assembly executor produces input and QKV/output/FFN weight gradients
matching the previously finite-difference-validated monolithic reference and
rejects stale contexts:

```sh
scripts/check_tensor_transformer_backward_executor_spike.sh
```

The first optimizer-loop spike trains the complete small encoder block on a
deterministic normalized sequence-pattern target. Every epoch runs the
21-action backward executor and updates QKV, output-projection, and both FFN
weight groups with SGD; the gate requires a substantial MSE reduction and
non-zero updates in every parameter group:

```sh
scripts/check_tensor_transformer_train_spike.sh
```

Everything above trains one architecture, so it doesn't yet show whether the
`{run, context, kind, flags}` assembly executor (`tensor_transformer_steps_execute`)
is actually general or only happens to work for this one encoder block. The
executor's own machine code has no attention/LayerNorm/FFN-specific logic in
it — it is a 32-byte step interpreter — so a second, unrelated architecture
through the same loop is the direct test. A 2→4→1 XOR MLP (unrelated math,
its own zero/reverse action set, its own `Context`/`emit`) is driven through
the identical executor object file. Its backward is checked twice — against
finite differences, then against a monolithic reference, exactly as the
encoder block was — and it trains from the same hand-picked starting weights
the existing `fasm/examples/xor_tensor_train.asm` uses, so a training-dynamics
difference can't be blamed on unlucky random init:

```sh
scripts/check_tensor_mlp_train_spike.sh
```

It converges to the exact XOR truth table (0, 1, 1, 0), matching that
existing trainer's locked result. The executor loop itself is confirmed
general; the planner/`emit()` logic that turns a graph into actions was still
hand-written per architecture at that point, not derived from a shared graph
compiler — `tensor_compiler_planner_check.c` already lowered arbitrary
matmul/bias/relu/mse node graphs (including this exact MLP shape) into fused
steps, but that lowering wasn't wired to numbers or an executor.

`tensor_graph_engine_check.c` closes that gap: the same op set (matmul,
bias, relu, mse) as real float buffers, with forward and backward both
dispatched generically by `node->op` through a small table — one function
per op, not one function per model — running through the exact same
`tensor_transformer_steps_execute` executor. The 2→4→1 XOR MLP this repo
now trains four different ways (`xor_tensor_train.asm`, the MLP-through-
transformer-executor spike above, `tensorctl mlp`, and this) is trained a
fourth time here with zero hand-written `forward()`/`backward()` for this
shape at all — only a 12-node graph description and a generic interpreter
that would describe any other matmul/bias/relu/mse graph identically.
Correctness is checked by finite differences on this exact engine, the same
oracle pattern every other spike in this repo uses:

```sh
scripts/check_tensor_graph_engine_spike.sh
```

Getting there caught a real bug on the first run: predictions diverged to a
flat ~-12.9 for every input. The zero-actions only reset the four trainable
parameters' gradient buffers, not the five intermediate nodes' — and
backward accumulates into every grad buffer with `+=`, so those five grew
across every training step forever instead of resetting each time. Fixing
that (and dropping the unused `d(loss)/d(x)` gradient buffer entirely,
rather than leaving it silently accumulating too) is what actually
converges. That bug also generalizes past this one graph: any live
gradient accumulator needs a defined initialization policy, or state leaks
across training steps — not an MLP quirk, a training-engine invariant.
Fusing steps the way the original planner spike's `STEP_MB`/`STEP_MBR` does
is a deliberately separate next refinement, not required for the core claim
here: this is one generic engine, not MLP-code plus Transformer-code that
both happen to call the same assembly function.

`tensor_graph_engine_transformer_check.c` is the real test of that claim,
not "add Transformer support": the same generic op-dispatch mechanism,
extended with four new op traits this block's real math needs —
`ATTENTION`, `RESIDUAL`, `LAYERNORM`, `CONTIGUOUS` — under a hard rule of no
`transformer_backward_emit()` or any transformer-specific compiler path.
`MATMUL`, `BIAS`, and `RELU` are reused completely unchanged from the MLP
engine (already generalized over rows/cols, so a `[T,*]` shape needed no
new code); only the four new ops are new, dispatched through the exact
same table. Same real shape as every other transformer spike here
(`T=3,M=4,H=2,D=2,F=6`); no `BIAS` nodes anywhere in this graph, because
the real Block has none in its FFN either (`matmul->relu->matmul`, no bias
add) — the graph simply doesn't use every op the engine knows about, which
is itself part of the point. Finite differences on all four trainable
groups (`wq,wo,w1,w2`) pass, and training converges from `loss=2.178741`
to `0.000000` — the same starting loss, to the last printed digit, as
every other transformer training spike in this repo, an independent
cross-check that the target generation and initial weights actually match
before a single gradient is compared:

```sh
scripts/check_tensor_graph_engine_transformer_spike.sh
```

Getting there caught one real, non-obvious setup gap, not a math bug: the
first run converged only to `loss=1.28`, plateaued, while finite
differences on the same run already passed — so the gradients were exact
and something else was wrong. The target wasn't per-row normalized like
`target_make()` elsewhere in this repo, and the graph's last op before the
loss is `LAYERNORM`, whose output is inherently zero-mean/unit-variance per
row — an unnormalized target has a floor a LayerNorm-terminated network
can't reach no matter how correct its gradients are. Matching the
established target convention is what actually gets to zero.

With generality proven on two very different graphs, the next real question
is whether an *optimization* pass on top of that graph — not just execution
— stays model-independent too, or starts needing per-model rules the moment
it tries to be clever. `tensor_graph_engine_fusion_check.c` answers it with
the literal case from `tensor_compiler_planner_check.c`: can one generic
pass see `MATMUL->BIAS->RELU`, decide (by consumer-count, not by which
model it's looking at) that it's safe to collapse, and replace three
scheduled actions with one — without ever bypassing the kernels already
proven correct, since the fused step just calls the same `k_matmul`/
`k_bias`/`k_relu` functions from inside one dispatch instead of three:

```sh
scripts/check_tensor_graph_engine_fusion_spike.sh
```

Run on the MLP graph, it finds exactly the grouping
`tensor_compiler_planner_check.c` already locked in statically — `MBR` then
`MB` then one plain step, 3 groups from 6 actions — and training still
converges to the same XOR truth table through the fused schedule. The
other half of the cross-check lives in
`tensor_graph_engine_transformer_check.c`: the identical, unmodified pass
run against the real Transformer graph, which has no `BIAS` nodes anywhere
(`N_Z1`, a matmul, feeds `N_ACT`, a relu, directly). The honest, correct
answer there is "nothing to fuse" — 12 groups, all plain — not a misfire
and not a special case carved out for this shape. Same code, two very
different graphs, two different but both *correct* outcomes: the
optimization pass turns out to be exactly as model-independent as the
execution path it sits on top of.

Every model this engine had trained up to this point — XOR, the T=3
Transformer block, GunPoint — was purpose-built for this repo. A user
pointing `tensorctl` at their own model brings neither the shapes nor the
data this project chose, so the real test is a model this repo didn't
design around at all. `tensor_graph_engine_mnist_check.c` is the first rung
of that ladder: the exact same 6-node matmul/bias/relu/matmul/bias/mse
graph as the XOR MLP — zero new op traits — but at real `784->32->10`
dimensions instead of `2->4->1`, on the real MNIST digit dataset (the same
`ossci-datasets` mirror `torchvision` uses), not a fixture built for this
repo:

```sh
MNIST_DIR=/path/to/extracted/mnist-idx-files \
  scripts/check_tensor_graph_engine_mnist_spike.sh
```

Without `MNIST_DIR`, the check skips (matching the NIR-MFCO/GunPoint
pattern). With it: the same fusion pass finds the identical `MBR`+`MB`
grouping as the tiny XOR graph — confirming the pass generalizes across
*size*, not just across models — and training on a 3,000-image subset for
8 epochs takes about a second and a half, moving test accuracy from 12.3%
(near the 10% chance floor) to 89.3%. Not a competitive MNIST result — the
point was never accuracy, it was proving the engine doesn't choke or need
new plumbing the moment real external data and non-toy dimensions show up.

### Reconciling two independently-built compilers

A parallel effort in this repo had, without coordination, built its own
semantic graph compiler (`tensor_semantic_training_compiler_check.c`) with a
genuinely reusable `compile()` entry point — automatic needs-grad
derivation, shape validation, and the optimizer step scheduled through the
executor like everything else — none of which `tensor_graph_engine_*.c`
above had. A structural comparison found the two were complementary layers,
not duplicates: Codex's file had the better compiler *skeleton*, this
repo's graph-engine files had the broader semantic *coverage* (the
Transformer op set, the `aux` saved-state mechanism, a real fusion pass, a
real external dataset). `tensor_semantic_compiler.h` merges them into one
canonical compiler, under a hard rule: no `forward_nodes[]`, `zeroable[]`,
or any other hand-enumerated per-model schedule list anywhere downstream —
a model is exactly a `Node` array, and `compile()` derives the rest
(needs-grad by forward reachability from `PARAM` leaves, zero-grad for
every node that needs one, backward in reverse order with an automatically
computed per-operand grad mask, optimizer steps for every trainable leaf —
all four still lowered to `ExecStep`s and run through
`tensor_transformer_steps_execute`, never a bare C loop). Op dispatch stays
table-driven (`FWD[op]`/`BWD[op]`) rather than Codex's if-chain, and
`ATTENTION`/`LAYERNORM` save their forward state through the generic `aux`
field on `Tensor` — there is no transformer-specific field anywhere in the
header. Fusion (`detect_fusion()`) is kept as a separate, optional pass
layered on top of `compile()`'s output, not logic built into `compile()` or
into any model's graph.

Three driver files prove the merge reproduces every previously-green result
through this one path, with nothing but a graph + tensors in each:

```sh
scripts/check_tensor_merged_mlp_spike.sh          # XOR: predictions=0,1,1,0, correct=4/4
scripts/check_tensor_merged_fusion_spike.sh        # same MBR+MB grouping, spliced onto compile()'s own schedule
scripts/check_tensor_merged_transformer_spike.sh   # loss=2.178741->0.000000, finite differences on wq/wo/w1/w2
MNIST_DIR=/path/to/extracted/mnist-idx-files \
  scripts/check_tensor_merged_mnist_spike.sh       # 784->32->10, test_accuracy=12.3%->89.3%
```

Both original implementations (`tensor_graph_engine_*.c` and
`tensor_semantic_training_compiler_check.c`) are left untouched and still
pass their own checks — kept as regression oracles for this merge, not
deleted alongside it.

### A new op trait, not a new compiler path: CONV

The cleanest test of the merge above is adding a model the canonical
compiler wasn't built around and checking what has to change.
`tensor_merged_cnn_check.c` adds `CONV` — a real valid, stride-1 2D
convolution, not an im2col-plus-matmul decomposition — to
`tensor_semantic_compiler.h`. The diff for that is one op-enum entry, one
forward/backward kernel pair, one dispatch-table entry, and one new branch
in `validate()`; `compile()`'s needs-grad derivation, zero-grad policy,
backward grad-mask computation, optimizer scheduling, and
`tensor_transformer_steps_execute` were not touched, and every
previously-green merged check (MLP, fusion, Transformer, MNIST) still
passes unmodified with `CONV` added to the shared header:

```sh
scripts/check_tensor_merged_cnn_spike.sh
```

Small synthetic task, same role XOR played before MLP ever saw real MNIST:
four fixed 6x6 single-channel images through one 3x3/2-filter conv (valid,
no padding) → relu → a small FC head, trained against four distinct fixed
targets. Conv output is declared as a flat `[1, HOUT*WOUT*COUT]` node
rather than `[HOUT*WOUT, COUT]` specifically so the very next `MATMUL`
consumes it with no reshape op in between — the memory layout is identical
either way, so this is a free reinterpretation, not new plumbing. Finite
differences pass on both the new conv weight and the pre-existing FC
weight; `steps=20` (`forward=5 zero=7 backward=5 optimizer=3`) is exactly
what `compile()`'s existing, unmodified algorithm derives for this graph;
loss goes `1.001362->0.000000` over 4000 epochs.

### CONV at real scale: MNIST-CNN vs MNIST-MLP through the same compiler

The synthetic fixture proved the architectural thesis; it didn't prove
`CONV` holds up at real scale and a real data distribution.
`tensor_merged_cnn_mnist_check.c` trains `conv(5x5,4 filters)->relu->fc->
bias->mse` on the same real MNIST digits and the same 3,000/1,000 train/
test subset `tensor_merged_mnist_check.c` (the MLP) already uses — same
canonical `compile()`, same executor, **no new op**: `CONV`'s output was
already declared as a flat `[1, HOUT*WOUT*COUT]` node, so the next
`MATMUL` reads it directly and no flatten/view trait was needed after all.
Finite differences on a real MNIST fixture (conv weight and FC weight)
pass before any training happens, then:

```sh
MNIST_DIR=/path/to/extracted/mnist-idx-files \
  scripts/check_tensor_merged_cnn_mnist_spike.sh
```

| | MLP (784→32→10) | CNN (conv 5×5×4→fc 2304→10) |
|---|---|---|
| compiled steps | 25 | **20** |
| declared graph bytes | 210,888 | 228,584 |
| epochs | 8 | 5 |
| wall time | 1.2s | 1.7s |
| test accuracy | 12.3%→89.3% | 8.8%→**91.4%** |

Two things worth being honest about rather than glossing over. First, the
CNN's *compiled step count is lower* than the MLP's (20 vs 25 — one `CONV`
action replaces what would otherwise be five nodes' worth of matmul-style
unrolling) while its *wall-clock cost per epoch is about 2.4x higher*
(`1.7s/5` vs `1.2s/8`) — because `CONV`'s nested nine-loop body
(`COUT×HOUT×WOUT×CIN×KH×KW`) runs entirely inside one `ExecStep`. Action
count and declared-buffer size are not proxies for compute cost once an op
like this exists; a planner (the cost-aware work earlier in this README)
that priced a schedule by step count or byte traffic alone would have
under-priced this graph. Second, the FC weight after an unpooled conv
(`2304×10`) ends up close in size to the MLP's whole first layer
(`784×32`) — removing pooling by design (`CONV` is the only new op; no
`POOL`) means the flatten-to-FC weight matrix, not the conv filters
themselves (`4×1×5×5` — tiny), is what actually dominates this graph's
memory, a genuine layout/lifetime consequence of the op set chosen, not
something either compile() or the executor had any say in.

### POOL: does the resource trade-off move the way it should?

The FC-after-flatten finding above pointed at an obvious next question, and
the parallel effort in this repo's differential-oracle work made the same
call independently: not another conv, but pooling — not for accuracy, but
to check whether shrinking the graph's intermediate width actually shrinks
FC parameters, memory, and compute the way it should. `POOL` (non-
overlapping max pool, stride == window) was added to
`tensor_semantic_compiler.h` under the exact same constraint as `CONV`: one
op-enum entry, one kernel pair, one dispatch-table entry, one line added to
`validate()`'s existing unary-op list (`POOL` joins `RELU`/`LAYERNORM`/
`CONTIGUOUS`/`ATTENTION` — ops with no `rhs`), one new `validate()` branch.
`compile()`'s needs-grad/zero-grad/backward-mask/optimizer logic was not
touched. Saved state (the argmax's flat input index per output element)
goes through the same generic `aux` mechanism `LAYERNORM`/`ATTENTION`
already use — not a new struct field.

```sh
scripts/check_tensor_merged_pool_spike.sh              # synthetic fixture, finite differences through the argmax route
MNIST_DIR=/path/to/extracted/mnist-idx-files \
  scripts/check_tensor_merged_cnn_pool_mnist_spike.sh   # real MNIST, conv->relu->pool->fc
```

`tensor_merged_pool_check.c` is the synthetic fixture (same role as
`tensor_merged_cnn_check.c` for `CONV`): the CONV fixture's exact 6x6x1 →
conv 3x3/2 filters → 4x4x2 graph, with a 2x2 max pool inserted before the
FC head (4x4x2 → 2x2x2). Finite differences on the conv weight only pass
if backward correctly routes gradient through the pool's per-output argmax
— confirmed. `tensor_merged_cnn_pool_mnist_check.c` is the real-scale
comparison: identical conv stage to `tensor_merged_cnn_mnist_check.c`
(5x5, 4 filters, 28x28x1→24x24x4), only difference a 2x2 pool before FC
(→12x12x4), same train/test subset, same printed fields:

| | MLP (784→32→10) | CNN, no pool (2304→10) | CNN, 2×2 pool (576→10) |
|---|---|---|---|
| compiled steps | 25 | 20 | 23 |
| declared graph bytes | 210,888 | 228,584 | **97,256** |
| wall time (5-8 epochs) | 1.2s | 1.7s | **1.0s** |
| test accuracy | 12.3%→89.3% | 8.8%→91.4% | 8.7%→**92.5%** |

The trade-off moved exactly the direction it should, on every axis at
once: pooling's 4x spatial downsample (2×2, stride 2) shrinks the FC
weight 4x (2304→576), which drags total declared graph memory down 2.4x
and wall-clock training time down ~1.7x relative to the unpooled CNN —
*and* accuracy went up, not down (91.4%→92.5%), consistent with pooling's
usual role as a mild regularizer rather than a pure compression trade.
None of this required touching `compile()`, `validate()` for any
pre-existing op, or the executor a second time — the same claim `CONV`
established, now confirmed by a second, independently-motivated op.

### One gate, three models: MLP vs CNN vs CNN+POOL

The comparisons above were three separate files printing adjacent numbers,
not one apples-to-apples run. `tensor_merged_mnist_resource_gate.c` is a
read-only consumer of `tensor_semantic_compiler.h` — no op or compiler work
happens here — that builds all three graphs, trains each under the
identical fixed `train_samples=3000 test_samples=1000 epochs=5` budget,
and reports, per model: compiled step count; declared graph bytes split
into parameter bytes (`PARAM` leaves) vs activation/saved-state bytes
(everything else); a breakdown of node count by op; a **static**
per-forward-pass MAC/element-op estimate computed purely from each node's
own declared shape and its operands' (no op-specific macros involved, so
it generalizes to any graph this header can compile); wall-clock training
time; and test accuracy:

```sh
MNIST_DIR=/path/to/extracted/mnist-idx-files \
  scripts/check_tensor_merged_mnist_resource_gate.sh
```

```
compiled_steps: cnn=20 pool=23 (pool has MORE steps: yes)
train_seconds:  cnn=1.78 pool=1.02 (pool is FASTER despite more steps: yes)
static_macs:    cnn=82955 pool=67979 (mac estimate correctly predicts pool is cheaper: yes)
```

This is the concrete negative example against a naive `cost = action_count`
planner: pooled CNN has *more* compiled steps than unpooled (23 vs 20) but
trains faster (~1.7x), so ranking by step count gets the trade-off
backwards. A cost estimate computed from shapes instead — `MATMUL` costs
`rows·cols_lhs·cols_rhs`, `CONV` costs `output_elements·kernel_volume`,
`POOL` costs roughly one visit per input element, everything else one op
per output element — ranks the two correctly (67,979 < 82,955) using
nothing but each node's own declared shape. The gate asserts this
ordering, not just prints it (`pool.steps > cnn.steps` *and*
`pool.train_seconds < cnn.train_seconds` *and* `pool.macs < cnn.macs` are
all hard-checked), so this negative result stays locked in as a regression
rather than a one-off observation. Clean under ASan+UBSan and `-Werror`,
same as every other merged check.

### From internal cost modeling to an external product benchmark

Everything above is internal: is this compiler general, and does its own
notion of cost track reality. A separate line of work in this repo
(`tensor_killer_mnist_native.c`, `scripts/tensor_killer_mnist.py`,
`scripts/tensor_killer_compare.py`) asked a different question at exactly
one model size (~25k params, the same 784→32→10 MLP this README trains
several other ways): compiled through the same canonical `compile()` path
to a native arm64 binary, batch=1, how does it actually compare to
ONNX Runtime and tinygrad on an M1 — not on throughput, but on cold
startup, peak RSS, and deploy footprint, the metrics that matter for a
small static model shipped as part of something else, not served behind a
warm process. The result at that one size was a real product hypothesis:
*ultralight runtime for small static CPU models, where startup/RSS/deploy
footprint matter more than warm latency* — startup 2.76ms vs ONNX
Runtime's 82.6ms, RSS 4.6MB vs 63.3MB, deploy 154KB vs 78.6MB, at the cost
of worse warm latency (30.6µs vs 6.5µs).

One size is a demo, not evidence of a product boundary. `tensor_killer_
scaling_native.c` generalizes the same native runner to any `[IN,HID,OUT]`
MLP at runtime (`MATMUL`/`BIAS_ADD`/`RELU` are already fully shape-generic
— `rows`/`cols` are `Node` fields, not compile-time macros — so this
needed zero changes to the compiler, op set, or executor; it's a read-only
consumer, same as the resource gate). `scripts/tensor_killer_scaling.py`
and `scripts/tensor_killer_scaling_matrix.py` sweep a family of six
same-architecture MLPs from ~25k to ~4M parameters (`IN=784`/`OUT=10`
fixed — same real MNIST input/output the killer spike already used, only
hidden width `HID` grows), each measured against ONNX Runtime and
tinygrad on cold startup, warm batch=1 latency, peak RSS, deploy
footprint, parameter count, and cross-engine correctness (all three
engines run byte-identical exported weights; their checksums and
accuracies must agree).

**Success criteria, fixed before running any measurement** (this matters —
it's easy to fit a threshold to a result after seeing it): niche survives
at size N iff, versus the best of `{onnxruntime, tinygrad}`, native is at
least 5x faster cold, at least 3x smaller peak RSS, *and* no worse than
10x slower warm. These are asserted in the script's own predeclared
constants, not chosen after the run:

```sh
MNIST_DIR=/path/to/extracted/mnist-idx-files \
  scripts/check_tensor_killer_scaling_matrix.sh
```

| params | HID | cold startup vs best | peak RSS vs best | warm latency vs best | niche survives |
|---|---|---|---|---|---|
| 25,450 | 32 | 31.7x faster | 13.3x smaller | 5.0x slower | **yes** |
| 100,180 | 126 | 28.1x faster | 12.8x smaller | 13.8x slower | no |
| 500,860 | 630 | 20.6x faster | 10.4x smaller | 37.3x slower | no |
| 1,000,120 | 1258 | 14.7x faster | 9.0x smaller | 44.3x slower | no |
| 2,000,230 | 2516 | 9.9x faster | 7.7x smaller | 45.5x slower | no |
| 3,998,860 | 5030 | 6.3x faster | 6.0x smaller | 18.0x slower | no |

Cross-engine correctness (checksum + accuracy agreement) held at every
size — the six points are all measuring the same thing, not diverging
implementations.

**Verdict: crossover between ~25k and ~100k parameters**, and it is a
narrower niche than the one-size demo implied — narrower than most
people's intuitive guess for "small static model." The cold-startup and
RSS advantages stay real and large across the *entire* measured range (6x
and 6x even at ~4M params, nowhere close to failing those two criteria on
their own); what kills the niche under these criteria is warm latency
alone, and it does so almost immediately after the first size. The most
likely mechanical reason, worth stating plainly rather than leaving as an
unexplained number: `tensor_semantic_compiler.h`'s `k_matmul_fwd` is a
plain scalar triple-nested loop with no SIMD and no threading, while
ONNX Runtime's matmul kernel is threaded and vectorized — so native warm
latency scales roughly linearly with parameter count at a materially worse
constant factor, and a compute-bound library-grade kernel pulls ahead
almost immediately once there's enough work to amortize its own overhead.
This isn't a compiler or op-set limitation (out of scope for this task
regardless — no compiler, graph-semantics, op-trait, or executor changes
were made to produce this matrix) — it's a property of the one reference
`MATMUL` kernel every model in this repo currently shares, and this repo
already has an existence proof that a NEON-optimized kernel measurably
beats scalar on this exact CPU (`tensor_kernel_dispatch_profile.c`, this
README's `tensorctl` section above) — swapping it in is a legitimate,
separate follow-up, not a claim this matrix makes for free.

### Chasing that follow-up: NEON made the dominant shape *worse*

The obvious next experiment: plug that existing, already-verified NEON
`matmul` into the canonical `MATMUL` kernel and rerun the same scaling
matrix under the same predeclared criteria, with the one variable isolated
— no threading yet, so a win or a loss is attributable to vectorization
alone. `tensor_matmul_scaling_profile.c` measures exactly that isolated
variable first, at the exact `(M=1,K,N)` shapes this workload actually
uses — not a generic square sweep — compiled plain `-O3` (the same flags
the killer benchmark actually builds with, not `-fno-vectorize`, so this
is "does hand NEON beat what the compiler already gives you for free in
the real build"):

```sh
scripts/check_tensor_matmul_scaling_profile.sh
```

```
hid=32    layer1 shape=1x784x32   scalar_ns=  4041.1  neon_ns=  9728.9  speedup=0.42x scalar_wins
hid=630   layer1 shape=1x784x630  scalar_ns= 39175.4  neon_ns= 69221.9  speedup=0.57x scalar_wins
hid=5030  layer1 shape=1x784x5030 scalar_ns=358258.7  neon_ns=540844.3  speedup=0.66x scalar_wins
hid=32    layer2 shape=1x32x10    scalar_ns=   167.8  neon_ns=   165.2  speedup=1.02x native_wins
hid=5030  layer2 shape=1x5030x10  scalar_ns= 25714.3  neon_ns= 25796.7  speedup=1.00x scalar_wins
```

On layer 1 — the dominant cost at every size (`K=784`, an order of
magnitude more work than layer 2's `K=HID,N=10`) — the existing NEON
kernel **loses to scalar `-O3` by 33-61%, consistently, at all six sizes**.
Layer 2 is a wash (±2%, noise). This isn't "barely moves" — deployed as-is
it would have made the killer benchmark's cold/warm numbers *worse*,
because the FLOP-dominant matmul got slower.

The mechanistic reason survives inspection: that NEON kernel's loop order
is `ikj`, vectorized over output columns — for each of the `K=784`
reduction steps it re-reads and re-writes the *entire* output row from
memory, because nothing keeps an accumulator resident in a register across
the `K` loop. That's memory-traffic-bound. The scalar `ijk` loop keeps one
accumulator in a register for the whole `K` reduction per output element
and only writes it once — cache-friendly by construction. The `ikj`
pattern's real advantage is amortizing that repeated output re-read/
re-write across *multiple* output rows sharing the same `A[i,p]` load —
which is exactly why `tensor_kernel_dispatch_profile.c`'s own matmul sweep
(above) tests `M∈{3,8,64}` and finds NEON winning there: this repo's
actual inference shape is `M=1` (batch=1), the one regime where that
kernel's design assumption doesn't hold.

Given that, the swap was **reverted**, not shipped: `k_matmul_fwd` in
`tensor_semantic_compiler.h` stays the plain scalar loop on every
platform, unchanged, with the reasoning kept in a comment at the call site
so the next person doesn't rediscover this by re-trying it. Every merged
check and the differential oracle were re-run after the revert and are
byte-for-byte back to their pre-experiment numbers — this was a clean,
reversible experiment, not left half-applied.

So: **the bottleneck is deeper than "no SIMD"**, which is itself a
concrete, useful answer, not a null result. It's specifically "the wrong
SIMD strategy for this shape" — a `K`-reduction kept in a register per
output element (rather than an output row re-touched per `K` step) is the
theoretically correct next design for `M=1`, but it needs a strided gather
of `B` (`B` is stored `[K,N]` row-major, so a fixed output column reads
`b[p*N+j]` for varying `p` — not contiguous), which is a materially
different kernel to design and verify, not a one-line swap — and isn't
implemented here.

### The register-resident redesign, tried anyway — and it lost harder

That "materially different kernel" got built and measured rather than
left as a paragraph. `tensor_matmul_gemv_profile.c` implements the
theoretically-correct design for `M=1`: block the output into 4-wide
column groups, and for *each* block run the entire `K` reduction with the
accumulator held in a NEON register (`vfmaq_f32`) the whole time, writing
that block's output exactly once at the end — sidestepping the strided-`B`
gather problem by blocking over `N` instead of vectorizing over `K`
directly. `B` is read exactly once in total (no re-scans), the output is
written exactly once in total (not `K` times, fixing the previous kernel's
actual problem), and `A` (784 floats, ~3KB) stays L1-resident across every
block's re-read of it. Verified numerically equivalent to
`tensor_kernel_dispatch_spike.h`'s `scalar_matmul` (same 3e-5 relative
tolerance that file's own `check()` already uses) at every shape below,
before any timing happened — and benchmarked with proper repetitions
(200) reporting median/min/max, not a single average:

```sh
scripts/check_tensor_matmul_gemv_profile.sh
```

```
hid=32    layer1 shape=1x784x32   scalar=10125.0[9333,10833]  gemv=15625.0[15083,18333]  0.65x scalar_wins
hid=630   layer1 shape=1x784x630  scalar=56208.3[55750,83667] gemv=151667[144083,210500] 0.37x scalar_wins
hid=5030  layer1 shape=1x784x5030 scalar=337209[322375,548167] gemv=1244542[1166542,1512833] 0.27x scalar_wins
dominant-shape (layer1) verdict: gemv_wins=0 scalar_wins=6 out_of=6 -> DO NOT SPECIALIZE
```

It lost *worse* than the first NEON attempt — down to 0.27x at the largest
size, versus the reverted kernel's 0.59-0.66x. That result forced the
actual question open: was the earlier "memory-traffic" diagnosis wrong, or
is something else going on? The profile script also runs both kernels
against a genuinely un-vectorized scalar baseline
(`-fno-vectorize -fno-slp-vectorize`, the same second build mode
`tensor_kernel_dispatch_profile.c` already established), and that
comparison flips completely: against *true* scalar, GEMV wins at **all
six** sizes (1.01-1.25x). The memory-traffic diagnosis was right; it just
wasn't the whole picture. What actually happened: the plain-C scalar loop
this repo compares against isn't scalar in the shipped `-O3` build — LLVM's
autovectorizer is already turning that trivial `s += a[p]*b[p*n+j]`
reduction into something that beats *both* hand-written NEON kernels tried
here, on this exact CPU, for this exact shape family. This repo already
has one precedent for exactly this trap
(`tensor_kernel_dispatch_profile.c`'s own header comment, and the
"SIMD auto-vectorization confound" this README's `tensorctl` history
records elsewhere) — it reproduced a second time, this time as the actual
decision-blocking result rather than a caught-in-review mistake.

Per the criteria fixed before this experiment — compare against
production `-O3`, not a disabled-vectorization baseline; specialize only
if it wins there — the answer is unambiguous: **do not specialize.**
`k_matmul_fwd` is untouched by this experiment (no dispatch, no `M=1`
branch, nothing shipped); the canonical compiler, graph semantics, and
planner policy were never in scope and stayed that way. The killer-
workload scaling matrix's crossover is therefore **unchanged: still
between ~25k and ~100k parameters** — there was no code change for it to
move in response to. Two independent hand-NEON designs have now both lost
to the compiler's own autovectorization of the reference C loop on this
target; the honest stopping point per the criteria set for this task is
here, not a third kernel attempt. If this gets revisited, the next
question isn't "try another hand kernel" — it's "why does clang's
autovectorizer beat straightforward hand NEON here," which is a compiler-
codegen question, not a numerical-kernel-design one.

### The metric in between: total time for a process's whole lifetime

Cold startup and warm latency leave a gap between them. Cold startup
(measured with `KILLER_SKIP_ACCURACY=1`, a single sample) doesn't really
include a real inference; warm latency is measured *after* discarding
startup and a warm-up call entirely — steady state only. Neither answers
the actual question for the product niche this README already
established (small, short-lived, static-model workloads): if a process
starts, does `N` inferences, and exits, which engine wins on *total*
wall-clock time, and how does that answer change with `N`?
`tensor_killer_lifetime_matrix.py` measures that directly — no new native
or Python inference code; `tensor_killer_mnist_native.c`'s `reps` argument
and `tensor_killer_mnist.py`'s `--reps` flag already control exactly this,
so this only varies that parameter and times the whole subprocess
externally, the same methodology `tensor_killer_compare.py`'s existing
cold-startup measurement already uses at `reps=1`:

```sh
MNIST_DIR=/path/to/extracted/mnist-idx-files \
  scripts/check_tensor_killer_lifetime_matrix.sh
```

| N inferences/process | native total | onnxruntime total | tinygrad total | winner |
|---|---|---|---|---|
| 1 | 4.35ms | 86.1ms | 563.7ms | native |
| 10 | 3.70ms | 87.5ms | 593.9ms | native |
| 100 | 6.48ms | 89.0ms | 1013.0ms | native |
| 1,000 | 37.4ms | 99.3ms | 3,939.1ms | native |
| 10,000 | 298.7ms | 202.0ms | 32,847.9ms | **onnxruntime** |

**Empirical crossover: between 1,000 and 10,000 inferences per process
lifetime** — native wins total wall-clock time below that range,
ONNX Runtime wins above it, at every `N` measured on either side. A rough
linear interpolation between those two points (native ≈29.4µs/inference
past its ~4.4ms floor; onnxruntime ≈11.6µs/inference past its ~86ms floor)
puts the crossover near `N≈4,500` — in the same order of magnitude as, if
not identical to, the back-of-envelope estimate (`N≈3,313`) computed from
the isolated cold-startup/warm-latency numbers alone, which is a useful
sanity check that the two measurement methods agree in direction even
though they're not measuring quite the same thing (the amortized-into-
`N` per-inference cost here is higher than the steady-state-only warm
number for both engines, since it includes each engine's own warm-up
transient — real cost a deployed process actually pays, not an artifact
to be measured away).

This is the sharper version of the product claim this README already
made: not "warm latency is 4.7x worse," but **"ONNX Runtime only starts
winning total execution time after roughly four to five thousand
inferences in a single process's lifetime — below that, for this exact
model, native wins on every axis, including the one ONNX Runtime is
supposed to own."** For the "start up, do a handful of inferences, exit"
shape of workload this niche was defined around, warm latency's isolated
4.7x deficit is close to irrelevant.

The memory planner (`tensor_transformer_liveness_check.c`, above) laid out a
real 31-buffer/35-event arena, but its one rematerialize-vs-save choice —
recompute attention scores instead of keeping them alive — was a policy
baked into the buffer list by hand, not a decision the planner made. A
follow-up spike turns it into an actual trade-off: lay out the same real
schedule twice, once with scores kept alive for their full need window and
once with the existing short recomputed interval, convert "bytes over a
memory budget" and "recompute cost" into one comparable cost, and pick
whichever side is cheaper. The remat layout reproduces the existing 4928-byte
arena exactly; the save layout costs 5440. The acceptance bar isn't "it
picked the same answer as before" — it's that the answer actually moves when
the inputs change: a generous budget prefers saving (recompute buys nothing),
today's tight budget prefers rematerializing (matching the existing hand
policy), and making recompute artificially expensive at that same tight
budget flips it back to saving. All three real-data outcomes, plus two
synthetic candidates with unrelated size/cost profiles, are asserted, not
just printed:

```sh
scripts/check_tensor_liveness_cost_planner_spike.sh
```

A follow-up asks the same "is this a real decision or a hardcoded one"
question about the optimizer, not the memory planner: today all four SGD
updates run as one batch strictly after all 21 backward actions finish, even
though the four weight gradients finish accumulating at very different
points inside that pass. Each gradient's real lifetime — first write (its
zero action) to last write (the `reverse_one()` case that finishes it, read
directly off that switch statement, not guessed) — is modeled the same way
as the memory planner's buffers, comparing today's deferred-to-the-end
policy against releasing each gradient right after its own last write. On
the real 31-buffer schedule this saves exactly 0 bytes, and that's a real,
mechanistic finding: the weight gradient and its co-produced activation
gradient come out of the same `mmback` call and are both consumed one tick
later, so there's no free tick between them regardless of optimizer timing,
and the arena is already saturated through that window. A second, isolated
case proves the mechanism itself is correctly implemented rather than
silently broken — a later consumer that doesn't share that same-tick
co-production constraint does reuse the freed space:

```sh
scripts/check_tensor_gradient_lifetime_planner_spike.sh
```

The third planner question is about layout, not memory: the QKV layout spike
already showed that merging attention heads `[H,T,D] -> [T,H*D]` is not a
legal zero-copy reshape, so the planner always inserts an explicit
`CONTIGUOUS` copy before the output projection matmul. Rather than building
a general layout-aware kernel dispatcher on spec, one concrete question is
tested first: does a kernel that reads the `[H,T,D]` layout directly, skipping
the copy, actually win? The reformulation turns out to be exact, not
approximate — for fixed head `h`, `head`'s `[T,D]` slice and `wo`'s `[D,M]`
row-slice are both already ordinary contiguous blocks (row-major storage
makes this fall out for free), so `proj = sum_h head_h[T,D] @ wo_h[D,M]` is
mathematically identical to (contiguous-copy + one `[T,M]x[M,M]` matmul) —
no strided/gather kernel is needed for this specific case, just H
accumulating matmuls over blocks that were contiguous all along:

```sh
scripts/check_tensor_merge_layout_profile_spike.sh
```

Correctness is checked exactly across 5 shapes first. Timed against true
scalar (the same `-fno-vectorize -fno-slp-vectorize` diagnostic build the
SIMD profiling spike needed to avoid being fooled by the compiler
auto-vectorizing the "baseline" side), skipping the copy wins at every shape
tested in isolation, including the real block's exact `T=3,H=2,D=2` — 2.62x
there, and 1.5-2.9x across token-count, head-dimension, and head-count
sweeps.

That isolated win does not survive being wired into the real block, though.
`tensor_merge_layout_train_check.c` builds a full layout-aware forward AND
backward for the real Block — correctness is exact (matches the monolithic
reference's output and all five gradients, plus an independent
finite-difference check) — but end-to-end timing against two honest
baselines (the monolithic `forward()`, which already fuses the merge write
into the attention loop for free and never actually pays a separate
CONTIGUOUS cost; and an explicit-contiguous variant that does pay it,
matching what a real planner-emitted action would cost) lands within
~1-2% either way, using median-of-15 trials to rule out the single-shot
timing noise that initially made two back-to-back process runs disagree on
which variant even won:

```sh
scripts/check_tensor_merge_layout_train_spike.sh
```

So, corrected: this is not the unconditional win it looked like in
isolation. The op-level speedup is real but gets diluted below the noise
floor by the rest of the forward pass (QKV projection, attention, two
LayerNorms, FFN) at this block's current tiny scale — the same lesson the
SIMD profiling spike already taught, now confirmed a second time on a
different optimization: an isolated microbenchmark does not predict
full-pipeline impact, in either direction. Of the three planner spikes,
none delivered an unconditional win on the current shape; all three did
deliver a real, honestly-measured answer.

The real-data follow-up reuses NIR-MFCO and converts each conveyor image into
three ordered bands with four false-colour/occupancy features per token. It
retains the repetition-zero test split, adds a learned physical-setup
embedding, and trains the transformer regression head through the 21-action
backward executor. Its MAE is reported beside the stronger global linear
calibration baseline; the transformer is not required to win:

```sh
NIR_MFCO_ARCHIVE=/path/to/NIR-MFCO-dataset-v1.0.0.zip \
  scripts/check_nir_mfco_transformer_spike.sh
```

An ordered/shuffled ablation on that same NIR-MFCO sequence fixture answers
whether the transformer benefits from real band order: a pooled (order-blind)
linear baseline, the ordered transformer, and a shuffled-band transformer all
land within noise of each other (2.6/2.8/2.7 MAE points), so this dataset's
band order carries no exploitable signal for the block; it remains a useful
linear/tensor regression fixture, not a case for the extra architecture.

GunPoint is the order-sensitive counterpart: a classic UCR univariate
time-series benchmark (50 train / 150 test, length 150, 2 classes) where
column order is a real hand-motion timeline, not an interchangeable label.
Each trace is z-normalized and pooled into the same `[3, 4]` shape as the
NIR-MFCO sequence spike (three temporal thirds, four summary stats per
third: mean/std/min/max), so the classification head reuses the existing
21-action backward executor unchanged. The archive stays external:

```sh
GUNPOINT_DIR=/path/to/extracted/GunPoint \
  scripts/check_gunpoint_spike.sh
```

(or `GUNPOINT_TRAIN=... GUNPOINT_TEST=...` for two explicit file paths).
Without either, the script runs only its deterministic self-test. The
self-test's harness check is telling on its own: two classes built from the
*same* three band values with only their order flipped are perfectly
separable by a position-aware linear model (100%) and sit at exact chance
for the pooled, order-blind one (50%) — confirming the harness actually
measures order sensitivity before any real data is involved. On the real
split, a pooled-band logistic baseline, an ordered-band logistic baseline, a
shuffled-band logistic baseline, and a small hand-derived recurrent baseline
are reported alongside the transformer's own ordered/shuffled runs, so the
architecture's usefulness on genuinely order-sensitive data is judged by the
same measurement, not by assumption.

Run against the real archive (`timeseriesclassification.com/aeon-toolkit`,
no password), the result is the opposite of NIR-MFCO's. The recurrent
baseline is sized to roughly match the transformer block's own parameter
count (113 vs ~117 weights) and trained for the same 2500 epochs, so
capacity and training budget aren't stacked against either model. Ordered
test accuracy climbs with model class — pooled logistic 56.7%, ordered
logistic 66.7%, recurrent 76.7%, transformer 95.3% — and every order-aware
model loses accuracy when its own bands are shuffled: logistic barely
(66.7%→64.7% shuffled), the recurrent baseline more (76.7%→66.0%), and the
transformer most of all (95.3%→74.0%, over 20 points on the same weights).
That is a real three-way answer to the questions this spike was built to
ask: order can't be ignored here (order-invariant pooling is the weakest
model), a plain sequential baseline already captures a good part of it, and
attention still adds a large, order-dependent margin on top of the
recurrent baseline rather than just re-deriving what pooling already knew.

| Command | Problem / approach | Output |
|---------|--------------------|--------|
| `best_time_to_buy_sell_stock.asm` | LC 121 via `dp.inc` | `5` |
| `binary_search.asm` | LC 704 binary search | `4` |
| `climbing_stairs.asm` | LC 70 via `dp.inc` | `8` |
| `contains_duplicate.asm` | LC 217 int hash map | `1` |
| `first_unique_character.asm` | LC 387 character counts | `0` |
| `house_robber.asm` | LC 198 via `dp.inc` | `4` |
| `implement_queue_using_stacks.asm` | LC 232 via `stack.inc` | `1 1 0` |
| `intersection_of_two_arrays.asm` | LC 349 int hash map | `2` |
| `invert_binary_tree.asm` | LC 226 via `tree.inc` | `4 7 9 6 2 3 1` |
| `linked_list_cycle.asm` | LC 141 via `listnode.inc` | `1` |
| `majority_element.asm` | LC 169 Boyer-Moore vote | `2` |
| `maximum_depth_binary_tree.asm` | LC 104 via `tree.inc` | `3` |
| `maximum_subarray.asm` | LC 53 via `dp.inc` | `6` |
| `merge_sorted_array.asm` | LC 88 two pointers from end | `1 2 2 3 5 6` |
| `merge_two_sorted_lists.asm` | LC 21 via `listnode.inc` | `1 1 2 3 4 4` |
| `middle_of_linked_list.asm` | LC 876 via `listnode.inc` | `3` |
| `missing_number.asm` | LC 268 xor | `2` |
| `move_zeroes.asm` | LC 283 in-place compaction | `1 3 12 0 0` |
| `nested_list_weight_sum.asm` | [LC 339](https://leetcode.com/problems/nested-list-weight-sum/) | `10` |
| `number_of_islands.asm` | LC 200 via `grid.inc` | `1` |
| `palindrome_linked_list.asm` | LC 234 via `listnode.inc` | `1` |
| `remove_duplicates_sorted_array.asm` | LC 26 in-place unique prefix | `5 0 1 2 3 4` |
| `reverse_linked_list.asm` | [LC 206](https://leetcode.com/problems/reverse-linked-list/) | `5 4 3 2 1` |
| `search_insert_position.asm` | LC 35 lower bound | `2` |
| `single_number.asm` | LC 136 xor | `4` |
| `sort_array.asm` | [LC 912](https://leetcode.com/problems/sort-an-array/) | `-1 0 1 2 3` |
| `two_sum.asm` | [LC 1](https://leetcode.com/problems/two-sum/) brute O(n²) | `0 1` |
| `two_sum_hashmap.asm` | LC 1 hash map O(n) | `0 1` |
| `valid_anagram.asm` | LC 242 character counts | `1` |
| `valid_parentheses.asm` | LC 20 via `stack.inc` | `1` |

Core libraries: [`base64.inc`](fasm/core/base64.inc), [`control.inc`](fasm/core/control.inc), [`dirwalk.inc`](fasm/core/dirwalk.inc), [`dp.inc`](fasm/core/dp.inc), [`file.inc`](fasm/core/file.inc), [`for.inc`](fasm/core/for.inc), [`grid.inc`](fasm/core/grid.inc), [`hashmap.inc`](fasm/core/hashmap.inc), [`hashmap_str.inc`](fasm/core/hashmap_str.inc), [`hex.inc`](fasm/core/hex.inc), [`json.inc`](fasm/core/json.inc), [`listnode.inc`](fasm/core/listnode.inc), [`macho.inc`](fasm/core/macho.inc), [`math_fp.inc`](fasm/core/math_fp.inc), [`mem.inc`](fasm/core/mem.inc), [`scanner.inc`](fasm/core/scanner.inc), [`search.inc`](fasm/core/search.inc), [`stack.inc`](fasm/core/stack.inc), [`tree.inc`](fasm/core/tree.inc), [`sort.inc`](fasm/core/sort.inc), [`str.inc`](fasm/core/str.inc), [`repl.inc`](fasm/core/repl.inc), [`oop.inc`](fasm/core/oop.inc) (vtable + methods).

OOP-style demo (`Playlist` with `append` / `print` / `reverse` via vtable):

```sh
fasm fasm/examples/oop_playlist.asm
arch -x86_64 ./fasm/examples/oop_playlist
```

Expected output: `3 1 4 1 5` then `5 1 4 1 3`.

## Mini-Redis REPL

In-memory string-key store with a Redis CLI–style REPL (stdin/stdout, no TCP):

```sh
fasm fasm/apps/miniredis.asm
arch -x86_64 ./fasm/apps/miniredis
```

Commands (v1.1): `PING`, `SET key value`, `GET key`, `EXISTS key`, `DEL key`, `DBSIZE`, `INCR key`, `DECR key`, `MGET k1 k2 …`, `KEYS`, `SAVE path`, `LOAD path`, `QUIT`, `EXIT`.

`SET` stores a signed int64 when the value parses as an integer; otherwise the raw token is stored as a string. `GET` prints ints, strings, or `(nil)` if missing. `INCR`/`DECR` require an int value (or create `1`/`-1` if the key is absent); strings return `ERR wrongtype`. `MGET` needs at least two key tokens (max 16 tokens per line). `KEYS` lists all keys (no pattern filter in v1). `SAVE`/`LOAD` use a line-oriented text dump (see below).

Limits: 256-byte lines, ASCII keys/values without spaces in v1, up to 16 tokens per REPL line.

Dump format (`# miniredis v1` header):

```text
# miniredis v1
I user 42
S greeting hello
```

`I` = int value, `S` = string value (remainder of line after the key token).

Example session:

```text
miniredis> SET user 42
OK
miniredis> SET greeting hello
OK
miniredis> GET user
42
miniredis> GET greeting
hello
miniredis> INCR user
OK
miniredis> GET user
43
miniredis> SAVE dump.txt
OK
miniredis> LOAD dump.txt
OK
miniredis> DEL user
1
miniredis> GET user
(nil)
```

Pipe a script:

```sh
printf 'SET a 1\nGET a\nDBSIZE\nQUIT\n' | arch -x86_64 ./fasm/apps/miniredis
```

Smoke tests: `fasm/tests/macos-smoke/str_hash.asm`, `hashmap_str.asm`, `repl_ping.asm`, `miniredis_script.asm`, `tcp_echo.asm`.

## fscan

Tiny literal-search CLI inspired by grep/ripgrep. It searches one or more files
for a byte-exact literal and prints matching lines as `path:line:text`.

```sh
fasm fasm/apps/fscan.asm
arch -x86_64 ./fasm/apps/fscan [-c] [-l] [-i] needle file.txt other.txt
```

Homebrew:

```sh
brew install kroq86/fasm-mac/fscan
fscan needle file.txt
```

Smoke test:

```sh
scripts/check_fscan.sh
```

## hexpeek

Tiny native hex dump CLI for peeking at file bytes. It demonstrates the
reusable [`hex.inc`](fasm/core/hex.inc) formatter plus chunked file reads.

```sh
fasm fasm/apps/hexpeek.asm
arch -x86_64 ./fasm/apps/hexpeek [-n bytes] [-s skip] file.bin
```

Homebrew:

```sh
brew install kroq86/fasm-mac/hexpeek
hexpeek -n 64 file.bin
```

Release packaging:

```sh
scripts/build-hexpeek-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_hexpeek.sh
```

## fmath

Tiny exact math CLI for rational numbers and polynomials. Coefficients are
passed low-to-high: `1 3 2` means `2x^2 + 3x + 1`.

```sh
fasm fasm/apps/fmath.asm
arch -x86_64 ./fasm/apps/fmath frac add 1/3 1/6
arch -x86_64 ./fasm/apps/fmath poly-derive 1 3 2
arch -x86_64 ./fasm/apps/fmath poly-integrate 1 3 2
arch -x86_64 ./fasm/apps/fmath poly-eval 2 1 3 2
```

Homebrew:

```sh
brew install kroq86/fasm-mac/fmath
fmath frac add 1/3 1/6
```

Release packaging:

```sh
scripts/build-fmath-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_fmath.sh
```

## raymaze

Tiny native raylib raycaster maze game. It is an original Doom-inspired mini
game, not a Doom port: it does not ship Doom code, WADs, maps, sprites, sounds,
names, or trademarks. It demonstrates the reusable fixed-point
[`raycast.inc`](fasm/core/raycast.inc) helpers plus the
[`ccall64.inc`](fasm/core/ccall64.inc) C ABI bridge for Mach-O objects.

```sh
brew install raylib
fasm --emit=macho-obj fasm/apps/raymaze.asm /tmp/raymaze.o
clang -arch x86_64 /tmp/raymaze.o $(pkg-config --cflags --libs raylib) -o raymaze
arch -x86_64 ./raymaze
```

Because current fasm-mac output is x86_64-only, the linked raylib must also be
x86_64. On Apple Silicon, the default `/opt/homebrew` raylib bottle is arm64;
use an Intel/Rosetta Homebrew raylib install for a real windowed build.

Snapshot mode for deterministic checks:

```sh
arch -x86_64 ./raymaze --snapshot snapshot.ppm
```

Release packaging:

```sh
scripts/build-raymaze-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_raymaze.sh
```

## httpmini

Single-threaded concurrent static HTTP server for macOS x86_64. It uses the
green-thread scheduler, `kqueue`, and nonblocking sockets: one slow or partial
client does not block other clients. Successful `GET` responses for regular
files use macOS `sendfile`.

```sh
fasm fasm/apps/httpmini.asm httpmini
arch -x86_64 ./httpmini --root ./public --port 8080 --bind 127.0.0.1
```

V1 serves local regular files with `GET` and `HEAD`, closes each connection
after one response, serves `/index.html` for `/`, writes a simple access log to
stderr, and rejects directories, symlinks, `%` escapes, backslashes, and `..`
path components.

Release packaging:

```sh
scripts/build-httpmini-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_httpmini.sh
```

## logbus

Kafka-like local durable append-only message broker for macOS x86_64. It uses
the green-thread scheduler, `kqueue`, nonblocking sockets, and a length-prefixed
record log with a message-offset index and CRC32C per record. Topic data is
stored as rotated base-offset segments plus a global offset index; tune segment
size with `--segment-bytes N`. Accepted `PRODUCE` writes are fsync-backed, and
restart recovery trims segment/index tails back to the committed global offset
index. v1.3 uses a breaking storage/raw fetch format:
`[u32_len][u32_crc32c][payload]`.

```sh
fasm fasm/apps/logbus.asm logbus
arch -x86_64 ./logbus --dir ./data --port 9092 --bind 127.0.0.1
```

V1 commands use a RESP-like protocol: `PING`, `PRODUCE topic payload`,
`FETCH topic offset max_bytes`, `FETCHBATCH topic offset max_bytes`,
`COMMIT group topic offset`, `OFFSET group topic`, and `QUIT`.
`FETCHBATCH` returns raw `[u32_len][u32_crc32c][payload]...` log bytes via macOS
`sendfile`. This is a local single-partition broker, not a distributed Kafka
replacement.

Release packaging:

```sh
scripts/build-logbus-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_logbus.sh
```

## logvec

Experimental brew-worthy tool: **batch snapshot** index builder plus exact
cosine top-k search. logbus stays dumb; FASM owns f32 dot/norm/top-k only;
Zig wires protocol, files, ingest, and doc_id mapping (C++ host available in
`fasm/apps/logvec/`). v0 metric: cosine similarity
(`score = dot / (norm(q)*norm(v))`, higher is better). `build-index`
is one-shot — it does not tail topics. Spec: [`docs/logvec.md`](docs/logvec.md).
System form (Level 4): [`docs/system_form.md`](docs/system_form.md).

```sh
fasm --emit=macho-obj fasm/apps/logvec_core.asm logvec_core.o
zig build-exe fasm/apps/logvec.zig logvec_core.o \
  -target x86_64-macos -mcpu=baseline -O ReleaseSafe -femit-bin=logvec
arch -x86_64 ./logvec search --index index.lv --query query.bin --top 5
```

C++ host (same CLI, binary name `logvec_cpp`):

```sh
fasm --emit=macho-obj fasm/apps/logvec_core.asm logvec_core.o
clang++ -std=c++20 -O2 -arch x86_64 \
  fasm/apps/logvec/logvec.cpp logvec_core.o -o logvec_cpp
arch -x86_64 ./logvec_cpp search --index index.lv --query query.bin --top 5
```

Smoke test:

```sh
scripts/check_logvec.sh
scripts/check_logvec_cpp.sh
scripts/bench_logvec.sh   # in-process top-k regression (1k/10k/100k × dim=768)
scripts/bench_perf.sh     # layered dot/topk/search/io + parallel + ragbox breakdown
```

v0.2 adds layered bench (`--layer dot|topk|search|io`), scalar vs AVX2 dot A/B,
parallel exact search (1–4 threads), and unit-vector top-k fast path. Exact
linear scan — ~4.5 ms for 10k×768 single-thread, ~1.4 ms with 4 threads (see
`docs/logvec.md`). Not ANN; agent-scale snapshots only.

## ragbox

Local-first codebase memory for AI agents: chunk a repo, embed via Ollama,
build a copyable `.lv` index + JSON manifest, and search it from the terminal.
One x86_64 binary — no Python venv, no vector DB server, no web platform. More:
[`docs/ragbox.md`](docs/ragbox.md). System form (Level 4):
[`docs/system_form.md`](docs/system_form.md).

Homebrew:

```sh
brew tap kroq86/fasm-mac https://github.com/kroq86/fasm-mac
brew install ragbox
brew install ollama
ollama pull nomic-embed-text
arch -x86_64 ragbox doctor --skip-ollama
arch -x86_64 ragbox build --root ./repo --out memory.lv
arch -x86_64 ragbox refresh --root ./repo --index memory.lv
arch -x86_64 ragbox search --index memory.lv --query "where is auth handled?" --json
```

Why not the obvious alternatives?

| Alternative | ragbox difference |
|-------------|-------------------|
| `ripgrep` | semantic search, not lexical search |
| vector DB server | copyable file snapshot, not a running service |
| RAG platform | local CLI for repo memory, not a web platform |

Manual build (from source):

```sh
fasm --emit=macho-obj fasm/apps/logvec_core.asm logvec_core.o
clang++ -std=c++20 -O2 -arch x86_64 -pthread \
  fasm/apps/ragbox/ragbox.cpp logvec_core.o -o ragbox
arch -x86_64 ./ragbox build --root ./repo --out memory.lv
arch -x86_64 ./ragbox refresh --root ./repo --index memory.lv
arch -x86_64 ./ragbox search --index memory.lv --query "auth middleware" --json
```

Release packaging:

```sh
scripts/build-ragbox-release.sh 0.3.0
scripts/check_ragbox_release.sh
```

Smoke test:

```sh
scripts/check_ragbox.sh
scripts/check_ragbox_release.sh
```

Optional live check (Ollama required):

```sh
scripts/check_ragbox_live.sh
```

## macdbg

AI-native LLDB snapshot debugger for macOS binaries. Its useful surface is the
CLI report mode: run a target once under LLDB batch mode and write a structured
JSON file with status, exit code, crash signal, registers, backtrace,
disassembly near the program counter, stack memory, Mach-O summary, and an
escaped LLDB output tail.

```sh
fasm --emit=macho-obj fasm/apps/macdbg.asm /tmp/macdbg.o
clang -arch x86_64 /tmp/macdbg.o $(pkg-config --cflags --libs raylib) -o macdbg
arch -x86_64 ./macdbg --snapshot ./program report.json
arch -x86_64 ./macdbg --snapshot --args ./program arg1 arg2 -- report.json
```

The report is intended to be consumed by tools and agents:

```json
{
  "tool": "macdbg",
  "mode": "snapshot",
  "status": "exited",
  "exit_code": 0,
  "signal": null,
  "registers": {"rip": "0x..."},
  "backtrace": [],
  "disasm": [],
  "stack_memory": []
}
```

There is also an experimental raylib snapshot viewer:

```sh
arch -x86_64 ./macdbg --ui ./program
arch -x86_64 ./macdbg --ui --args ./program arg1 arg2
```

The UI is not a live step debugger yet. Press `R` to rerun the LLDB snapshot,
`J` to toggle the raw JSON/LLDB tail view, and `Esc` to quit. Because current
fasm-mac output is x86_64-only, the linked raylib must also be x86_64, just
like `raymaze`.

Release packaging:

```sh
scripts/build-macdbg-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_macdbg.sh
```

## pathsum

Tiny native recursive directory counter. It demonstrates the reusable
[`dirwalk.inc`](fasm/core/dirwalk.inc) API for directory traversal, file type
detection, and `stat64` size reads.

```sh
fasm fasm/apps/pathsum.asm
arch -x86_64 ./fasm/apps/pathsum [dir]
```

Output:

```text
files 2
dirs 1
bytes 1234
```

Homebrew:

```sh
brew install kroq86/fasm-mac/pathsum
pathsum .
```

Release packaging:

```sh
scripts/build-pathsum-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_pathsum.sh
```

## setdb

Tiny pure set-theoretic database CLI. A database is a `universe.db` directory
with an append-only operation log; the model is only sets and binary relations:
no SQL, no NULL, no duplicate rows, and no multisets.
Query results use the reusable [`arena.inc`](fasm/core/arena.inc) region
allocator: a command allocates temporary set/relation results in one arena, then
the process exits and the whole invocation lifetime is reclaimed at once.

```sh
fasm fasm/apps/setdb.asm setdb
arch -x86_64 ./setdb new universe.db
arch -x86_64 ./setdb add universe.db users alice bob carol
arch -x86_64 ./setdb add universe.db admins alice
arch -x86_64 ./setdb relation universe.db follows alice bob
arch -x86_64 ./setdb relation universe.db follows bob carol
arch -x86_64 ./setdb relation universe.db follows carol dana
arch -x86_64 ./setdb diff universe.db users admins
arch -x86_64 ./setdb select universe.db follows first alice
arch -x86_64 ./setdb join universe.db follows follows
arch -x86_64 ./setdb domain universe.db follows
arch -x86_64 ./setdb range universe.db follows
arch -x86_64 ./setdb inverse universe.db follows
arch -x86_64 ./setdb transitive-closure universe.db follows
arch -x86_64 ./setdb sets universe.db
arch -x86_64 ./setdb relations universe.db
arch -x86_64 ./setdb contains universe.db alice
arch -x86_64 ./setdb pairs universe.db follows
```

Tag-sugar layer (`tag`/`files`/`tags`) is a thin convenience wrapper over
`add`/`relation`/`select`, fixed to the sets `files`/`tags` and relation
`has_tag`:

```sh
arch -x86_64 ./setdb new files.db
arch -x86_64 ./setdb tag files.db song1.mp3 music jazz
arch -x86_64 ./setdb tag files.db song2.mp3 music
arch -x86_64 ./setdb files files.db music
arch -x86_64 ./setdb tags files.db song1.mp3
```

Names may contain letters, digits, `_`, `-`, `.`, and `/`, so real filesystem
paths work as atoms directly. `store-domain`/`store-range`/`store-inverse`
persist a query result as a named set or relation (an `SADD`/`RADD` per
result row) instead of only printing it, so it can feed a later query —
`domain`/`range`/`inverse` alone only ever print to stdout:

```sh
arch -x86_64 ./setdb store-domain universe.db follows leaders
arch -x86_64 ./setdb diff universe.db users leaders
```

`load` bulk-applies facts from a file instead of one `setdb` process per
fact — each line is `SADD`/`SREM`/`RADD`/`RREM`, the same wire format
already used for `ops.log`; `#` and blank lines are ignored, and a bad
line stops the load without rolling back lines already applied. `dump`
prints the current state back out in that same format, so `dump | load`
round-trips:

```sh
arch -x86_64 ./setdb load universe.db data/fasm_mac_readiness.setdb
arch -x86_64 ./setdb dump universe.db > snapshot.setdb
arch -x86_64 ./setdb new copy.db
arch -x86_64 ./setdb load copy.db snapshot.setdb
```

`scripts/dogfood_setdb_fasm_mac.sh` uses `data/fasm_mac_readiness.setdb`
to self-audit this repo with setdb: which apps lack a check/release
script or Homebrew formula, which core headers have no app consumer.

Output examples:

```text
bob
carol
```

```text
(alice,carol)
(bob,dana)
```

```text
(alice,bob)
(alice,carol)
(alice,dana)
(bob,carol)
(bob,dana)
(carol,dana)
```

```text
admins
users
```

```text
song1.mp3
song2.mp3
```

```text
jazz
music
```

Homebrew:

```sh
brew install kroq86/fasm-mac/setdb
setdb new universe.db
```

Release packaging:

```sh
scripts/build-setdb-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_setdb.sh
```

## machodoctor

Standalone macOS Mach-O inspector intended to ship as its own Homebrew formula
and ready-to-run binary. It is separate from `fasm-mac`; FASM is only used to
build release artifacts. Universal/fat Mach-O files are supported by inspecting
their x86_64 slice in v1.

```sh
fasm fasm/apps/machodoctor.asm
arch -x86_64 ./fasm/apps/machodoctor ./fasm/apps/machodoctor
arch -x86_64 ./fasm/apps/machodoctor --json ./fasm/apps/machodoctor
arch -x86_64 ./fasm/apps/machodoctor --deps ./fasm/apps/machodoctor
arch -x86_64 ./fasm/apps/machodoctor --check ./fasm/apps/machodoctor
```

Release packaging:

```sh
scripts/build-machodoctor-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_machodoctor.sh
```

## shipcheck

Standalone local release QA checker for Homebrew-style binary products. It
validates one formula, one release tarball, and one built Mach-O binary before
uploading release assets.

```sh
fasm fasm/apps/shipcheck.asm
arch -x86_64 ./fasm/apps/shipcheck Formula/hexpeek.rb dist/hexpeek-0.1.0-macos-x86_64.tar.gz ./hexpeek
```

Checks include formula `url` basename, `version`, `sha256`, `bin.install`, the
tarball filename shape, and whether the binary is an x86_64 Mach-O executable.
Tar archive contents remain a shell-script smoke check in v1.

Homebrew:

```sh
brew install kroq86/fasm-mac/shipcheck
shipcheck Formula/hexpeek.rb dist/hexpeek-0.1.0-macos-x86_64.tar.gz ./hexpeek
```

Release packaging:

```sh
scripts/build-shipcheck-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_shipcheck.sh
```

## logknife

Tiny structured log slicer for plain logs and JSONL. It is the first consumer
of the reusable line scanner in `fasm/core/scanner.inc` and the minimal JSONL
field matcher in `fasm/core/json.inc`.

```sh
fasm fasm/apps/logknife.asm
arch -x86_64 ./fasm/apps/logknife --contains timeout app.log
arch -x86_64 ./fasm/apps/logknife --jsonl --level error --count app.jsonl
arch -x86_64 ./fasm/apps/logknife --jsonl --field status=500 app.jsonl
```

Release packaging:

```sh
scripts/build-logknife-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_logknife.sh
```

## Mini-Redis TCP server (v0)

RESP/TCP server on port **6379** (macOS x86_64, one client at a time):

```sh
fasm fasm/apps/miniredis_server.asm
arch -x86_64 ./fasm/apps/miniredis_server
```

Commands: `PING`, `SET`, `GET`, `QUIT` (same semantics as REPL for values: int if parseable, else string).

Test with `redis-cli` (x86_64/Rosetta) or `nc`:

```sh
redis-cli -p 6379 PING
redis-cli -p 6379 SET foo bar
redis-cli -p 6379 GET foo
redis-cli -p 6379 QUIT
```

Without redis-cli:

```sh
printf '*1\r\n$4\r\nPING\r\n' | nc localhost 6379
```

TCP echo smoke (port 9999):

```sh
fasm fasm/tests/macos-smoke/tcp_echo.asm
arch -x86_64 ./fasm/tests/macos-smoke/tcp_echo &
printf 'hi' | nc localhost 9999
```

```sh
scripts/check_leetcode_examples.sh
```

## Shared Libraries

Build a Mach-O object and link it into a `.dylib`:

```sh
fasm --emit=macho-obj add.asm add.o
clang -arch x86_64 -dynamiclib wrapper.c add.o -o mylib.dylib
```

On Apple Silicon, Python `ctypes` examples need an x86_64/Rosetta Python to
load that dylib. An arm64 Python cannot load an x86_64 library.

## Current Limits

- Output is x86_64 only.
- Native arm64 fasm classic is out of scope.
- The executable converter supports simple ELF64 executable layouts.
- The object converter supports simple allocatable `.text`, `.data`, `.bss`
  sections and symbols.
- ELF relocations in object files are rejected for now.
- fasm classic still does not understand `format Mach-O`.
- Coroutines and other callback/stack-switching examples need separate ABI
  review before being called supported on macOS.

## Build The Host

The checked-in bridge expects the macOS x64 host binary at:

```text
fasm/build/out/macos-x64/fasm-macos-x64
```

Rebuild it with:

```sh
./fasm/build/macos-x64.sh
```

Verify:

```sh
file fasm/build/out/macos-x64/fasm-macos-x64
arch -x86_64 fasm/build/out/macos-x64/fasm-macos-x64
```

## Smoke Tests

```sh
fasm fasm/basic/fib.asm
arch -x86_64 ./fasm/basic/fib

fasm --emit=elf fasm/basic/fib.asm /tmp/fib.elf
file /tmp/fib.elf

fasm --emit=macho-obj /path/to/add.asm /tmp/add.o
file /tmp/add.o
```

## Upstream

- Original project: <https://flatassembler.net/>
- Upstream archive used here: `fasm-1.73.35.tgz`

The original license is kept at [fasm/license.txt](fasm/license.txt).

## References

- <https://flatassembler.net/> — flat assembler (fasm1) by Tomasz Grysztar
- <https://2ton.com.au/> — HeavyThing x86_64 FASM library by Jeff Marrison (GPLv2+); algorithms in `mem.inc`, `base64.inc`, `math_fp.inc` adapted from here
- <https://www.agner.org/optimize/> — Agner Fog's optimization guides and asmlib; small-copy and memcmp patterns
- <https://board.flatassembler.net/> — FASM community board; macro techniques and fasm1 idioms
- <https://github.com/tgrysztar/fasmg> — fasmg by Tomasz Grysztar; `control.inc` structured-flow macros inspired by `packages/x86/include/macro/if.inc`
