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
general; the planner/`emit()` logic that turns a graph into actions is still
hand-written per architecture, not derived from a shared graph compiler — the
recurring `tensor_compiler_planner_check.c` spike already lowers arbitrary
matmul/bias/relu/mse node graphs (including this exact MLP shape) into fused
steps, but that lowering isn't wired to either executor yet.

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
