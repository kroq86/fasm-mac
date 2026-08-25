# Autograd architecture spikes

The scalar tape in `fasm/core/autograd_f32.inc` is a correctness spike. It is
useful for checking derivatives of small expressions, but it is not yet the
storage model for trainable dense networks.

## Evidence from the Gazes implementation

The author's description explicitly calls out AVX2 and FMA and says that large
matrix multiplications are processed in chunks. The source agrees: multiple
`float32` products are accumulated in YMM registers. This is a useful kernel
technique, but the original code hard-codes chunk widths into particular layer
loops. A reusable implementation needs a scalar remainder path and runtime
dispatch rather than layer-size requirements.

## Spike 1: scalar graph capacity

For one dense layer with `I` inputs and `O` outputs, a literal scalar graph
needs roughly `2 * I * O + O` operation nodes: one multiply and one add for
each weight, plus an activation per output. At 32 bytes per current node:

| Layer | Operation nodes | Tape bytes, excluding leaves |
|---|---:|---:|
| 2 x 2 | 10 | 320 B |
| 784 x 128 | 200,832 | 6,426,624 B |
| 1024 x 1024 | 2,098,176 | 67,141,632 B |

Batching multiplies this temporary graph again. Therefore scalar expansion is
acceptable as a derivative oracle and teaching implementation, but tensor
operations must be single graph nodes in the training runtime.

## Spike 2: pointers versus indices

The current scalar node stores direct parent pointers. This makes a fixed,
caller-owned tape simple, but growing or relocating the tape invalidates every
edge. A production tape should use node indices or stable arena pages.

Decision to test next: use 32-bit parent indices. They permit relocation, cap a
single tape at about four billion nodes, reduce edge storage, and make bounds
validation possible. Stable arena pages remain an alternative if profiling
shows index-to-address conversion is material.

## Spike 3: gradient lifecycle

The current `backward` clears gradients before every pass. Training needs these
as separate operations:

1. `zero_grad(parameters)` at optimizer-controlled boundaries.
2. `backward(loss)` that accumulates gradients.
3. `step(parameters, learning_rate)`.

This is required for mini-batch accumulation, shared parameters, and multiple
loss terms. The scalar spike must test shared subgraphs and repeated operands
before its behavior is used as the tensor reference.

## Spike 4: tensor operation boundary

The first tensor tape prototype should support only:

- caller-owned contiguous `f32` storage;
- explicit rank and dimensions with overflow-checked element counts;
- `matmul`, bias add, ReLU, and mean squared error;
- one node per tensor operation;
- backward kernels writing into caller-owned gradient buffers;
- scalar kernels first, with AVX2/FMA selected only after feature detection;
- non-multiple vector widths handled by a scalar tail.

Dense/MLP product wiring stays outside core. SIMD is an implementation detail
of `matmul`, not a constraint on model dimensions.

## Gates before XOR training

- Gradient-check each operation against finite differences.
- Test a shared parameter used by two paths.
- Test two backward calls with and without `zero_grad`.
- Test dimensions `1`, `7`, `8`, `9`, and a non-square matrix.
- Reject shape mismatch and multiplication overflow without writing outputs.
- Compare scalar and AVX2/FMA kernels within a documented float tolerance.
- Record peak tape and tensor memory for the XOR model and a 784 x 128 layer.

Only after these gates pass should the tensor API be treated as stable enough
for loss functions, SGD, and the XOR training example.

## Tensor matmul spike

`fasm/spikes/tensor_matmul_f32.asm` keeps an experimental scalar tensor kernel
outside core. Its C harness checks forward and both backward gradients against
central finite differences. Cases cover inner dimensions 1, 7, 8, and 9 plus
non-square output and shape rejection. It also verifies two backward calls
accumulate and that one tensor used as both operands receives both gradient
contributions. The backward kernel never clears caller-owned gradients.

```sh
scripts/check_tensor_matmul_spike.sh
```

This settles the mathematical operation boundary, but not the tape descriptor,
shape ownership, overflow policy, or SIMD dispatch API.

## Indexed tape and shape-validation spike

`fasm/spikes/tensor_tape_f32.asm` tests a 32-byte node with 32-bit parent
indices. Validation requires parents to precede their consumer, rejects unknown
operations and malformed leaf/unary/binary edges, and continues to pass after
the entire tape is copied to a different address.

Tensor descriptor validation rejects null/empty storage and distinguishes bad
descriptors (`-1`) from overflow (`-2`). Both `rows * cols` and the resulting
`float32` byte count must fit in `uint64` before a buffer may be addressed.

```sh
scripts/check_tensor_tape_spike.sh
```

The result favors indices over intra-tape pointers. Tensor data remains an
external view in this spike; ownership and lifetime are deliberately deferred.

## End-to-end tensor pipeline spike

The indexed tape now executes `matmul -> ReLU -> MSE` forward and in reverse.
The harness finite-difference checks both matmul parameter tensors through the
complete graph and verifies a second backward call accumulates gradients.

```sh
scripts/check_tensor_pipeline_spike.sh
```

No operation-specific saved context was required for these three operations:
their backward kernels use parent/output tensors already referenced by the
tape. This supports the current 32-byte node shape. More complex operations may
still require an optional context index rather than widening every node.

The repeated-backward test exposed an important lifecycle rule: retaining every
gradient causes intermediate-node gradients to be propagated again, producing
super-linear accumulation (`4x` on the second pass in the test graph). Before
each backward pass the executor now clears gradients of non-leaf intermediates,
keeps the caller-provided final-output seed, and preserves leaf gradients for
linear parameter accumulation. A later parameter/constant flag can control
which leaf gradients optimizers retain or ignore.

## Arena ownership and lifetime spike

`fasm/spikes/tensor_arena_f32.asm` separates persistent parameter storage from
resettable scratch storage. Both use caller-provided buffers and return 64-byte
aligned, zero-filled `float32` data/gradient regions. Resetting scratch bumps an
arena generation; owned tensor views carry the generation at allocation time,
so validation returns `-4` for stale views even when their old address has been
reused by a later allocation.

```sh
scripts/check_tensor_arena_spike.sh
```

This catches use-after-reset but expands an owned tensor descriptor from 32 to
48 bytes. The spike must compare this cost against storing owner/generation in
tape-side metadata before the public tensor ABI is frozen.

Tape flags now require every leaf to be exactly one of `parameter`, `constant`,
or `input`; operation results must be `temporary`. Invalid combinations are
rejected during topology validation. This supplies explicit policy inputs for
the optimizer and future gradient allocation without encoding them indirectly
through null pointers or opcodes.

## SIMD dispatch spike

`fasm/spikes/tensor_matmul_simd_f32.asm` preserves the scalar matmul ABI while
adding an AVX2/FMA implementation and automatic runtime selection. Dispatch
requires CPU AVX2/FMA bits, OSXSAVE/AVX, and XCR0 XMM/YMM state; otherwise it
uses the scalar kernel and never executes an unsupported instruction.

The vector kernel processes eight adjacent output columns and has a scalar FMA
tail. Shapes with widths 1, 7, 8, and 9 plus non-square 7x9/9x7 cases are
compared against the scalar reference within an explicit float tolerance.
The current arm64/Rosetta test host selects the scalar fallback, so the AVX2
object is assembled and linked but its explicit instruction path still needs a
run on AVX2/FMA-capable Intel hardware before promotion from spike status.

```sh
scripts/check_tensor_simd_spike.sh
```

SIMD is therefore a replaceable kernel detail, not a model-shape constraint.

The same dispatch now covers matmul backward. AVX2/FMA computes both
`dA += dOut @ transpose(B)` and `dB += transpose(A) @ dOut`, preserving
pre-existing gradients and using scalar tails for widths not divisible by
eight. The harness compares both gradients with the scalar reference for the
same 1/7/8/9 and non-square cases. On the current Rosetta host this exercises
the auto-dispatch fallback; explicit SIMD execution remains an Intel gate.

## Lifetime representation comparison

A 32-byte `TensorView` contains no identity beyond its addresses and shape. If
scratch storage is reset and reused for the same shape, stale and fresh views
are byte-for-byte identical; arena metadata alone cannot distinguish them.
A 48-byte owned descriptor can always carry owner/generation. Alternatively,
the existing eight reserved node bytes can hold a 32-bit arena slot and 32-bit
generation, preserving both a 32-byte view and 32-byte node while providing
stale detection only when the view is accessed through its tape node.

```sh
scripts/check_tensor_lifetime_spike.sh
```

The likely split is therefore a small public `TensorView` for kernels and a
tape-side lifetime token for graph execution. Standalone long-lived owned views
would still need a separate checked handle.

The executor ABI now receives an arena table and validates every node's packed
`arena_slot:generation` token before the first kernel call. A stale scratch
generation returns `-4`; the pipeline harness verifies the destination remains
unchanged, proving validation occurs before partial execution.

## Parameter iteration and SGD spike

`fasm/spikes/tensor_sgd_f32.asm` iterates only nodes flagged `parameter`, applies
`data -= learning_rate * grad`, and exposes parameter-only `zero_grad`.
It preflights all parameters before any write and rejects duplicate parameter
descriptors, preventing a shared tensor from being stepped twice or partially
updated before an error. Graph builders should represent shared parameters as
one leaf referenced by multiple consumers.

```sh
scripts/check_tensor_sgd_spike.sh
```

## Bias-add spike

Tensor bias-add treats an `MxN` input and `1xN` bias as one graph operation.
Backward accumulates the output gradient elementwise into the input and reduces
rows into the bias gradient. Central finite differences cover widths 1, 7, 8,
and 9, and invalid bias shapes are rejected.

```sh
scripts/check_tensor_bias_spike.sh
```

## Unified regression gate

All scalar-reference, finite-difference, topology, lifetime, arena, optimizer,
pipeline, and SIMD-dispatch spikes run through one gate:

```sh
scripts/check_tensor_runtime_spikes.sh
```

The AVX2/FMA object is always assembled and linked. Execution of its explicit
path remains a separate hardware gate when runtime feature detection reports it.

## ABI promotion and XOR gate

Hardware-independent scalar contracts now live in focused
`fasm/core/tensor_*_f32.inc` modules; spike sources are thin object wrappers.
`fasm/core/tensor_runtime_f32.inc` fixes the minimal 32-byte TensorView and
TapeNode layouts, opcodes, flags, index sentinel, and return-code meanings.

The `fasm/examples/xor_tensor_train.asm` consumer builds a real
`Linear -> ReLU -> Linear -> MSE` tape, runs forward/backward, parameter
zeroing, and SGD for 1,500 epochs, then gates all four XOR classifications.

```sh
scripts/check_xor_tensor.sh
```

## Compiled execution-plan spike

`fasm/spikes/tensor_plan_f32.asm` lowers validated semantic nodes into 40-byte
direct-call steps containing resolved forward/backward function pointers and
TensorView operands. Leaves disappear from the execution stream; an epoch no
longer resolves parent indices or switches on opcodes. The initial spike keeps
lifetime validation and gradient preparation outside the plan deliberately.

```sh
scripts/check_tensor_plan_spike.sh
```

## Edge-level selective differentiation spike

Gradient need is propagated from trainable leaves through each input edge. A
frozen `W2` in `H @ W2` still requires `dH` but not `dW2`, while the first
`X @ W1` requires `dW1` but not `dX`. Thus one boolean per operation is
insufficient: binary steps need lhs/rhs masks. A direct-call plan can avoid
widening PlanStep by selecting one of lhs-only, rhs-only, or both backward
entrypoints at compile time. When every parameter is frozen the compiled
backward plan is empty.

```sh
scripts/check_tensor_grad_prune_spike.sh
```

## Static scratch-liveness spike

The XOR planner scans an explicit combined schedule of six forward and six
reverse events. Tensor data and gradient buffers have independent intervals;
for example `z1.data` lives at events 0-1 while `z1.grad` lives at 10-11.
Inclusive interval coloring gives a 256-byte theoretical lower bound (four
64-byte slots). Intervals must be processed by first use; declaration order
produced a needlessly large five-slot plan in the initial spike. The executable
plan must additionally respect the current kernels' accumulation contract:
backward destinations use `grad += ...`, so a recycled gradient slot requires
an explicit zero-init step at the start of its new lifetime. That step is not in
the plan ABI yet. The safe plan therefore reuses saved-data ranges but keeps the
five gradient ranges unique: data peak 160 bytes plus 224 gradient bytes equals
384 bytes. The raw all-buffer live peak is 208 bytes.

```sh
scripts/check_tensor_liveness_spike.sh
```

The XOR executable now applies the conservative offsets to one real 384-byte
scratch block. Automatic lowering recognizes both single-consumer chains and
executes three direct-call steps (`MBR`, `MB`, `MSE`) instead of six. Its eleven
former distinct buffers contain 452 payload bytes (704 bytes if every allocation
is independently rounded to 64 bytes), so the currently executable reuse saves
68 payload bytes, about 15%. Reaching the 256-byte lower bound requires compiler-
inserted zero-init steps or overwrite-mode backward kernels. Plan integration
also required a unary ReLU thunk so every direct step can use the common
`(lhs, rhs, out)` call shape.

## Execution-fusion kernel spike

`matmul+bias` and `matmul+bias+ReLU` now have common six-pointer forward and
backward kernel ABIs and are checked against execution of the original scalar
operations. The semantic operations and their intermediate tensors remain
present; fusion is only an alternative compiled execution step.

The existing 40-byte `PlanStep` is unchanged. A fused step stores a pointer to
a 48-byte `FusionContext` in its `lhs` field; a thunk expands that context into
`A`, `W`, bias, saved matmul output, saved bias output, and final output. The
context is owned by and must live as long as the compiled plan. The spike proves
three semantic operations execute as one PlanStep with identical forward values
and gradients. Automatic pattern recognition and single-consumer validation are
now used by the XOR runtime compiler.

```sh
scripts/check_tensor_fusion_spike.sh
```

## Unified compiler-planner spike

The XOR semantic graph is also passed through a traits-driven planner. Its
`op_traits` table supplies arity, backward save requirements, and whether an
operation may be rematerialized. Consumer counting permits the first
`matmul+bias+ReLU` and second `matmul+bias` chains to lower from six semantic
operations to three execution steps, while an injected side consumer correctly
blocks the affected fusion.

The same pass produces nine edge-level gradient requirements, per-step save
masks, one explicit rematerialization action that trades a saved 64-byte
pre-ReLU value for recomputation, and four zero-init actions required by the
256-byte scratch lower-bound plan. Freezing every parameter produces an empty
backward plan. The assembly XOR executable now consumes the fused contexts and
locks the former unfused result (`loss=0`, predictions `0 999 999 0`) as its
exact regression baseline. Emission of zero-init and rematerialization action
kinds remains necessary before switching runtime scratch from 384 to 256 bytes.

```sh
scripts/check_tensor_compiler_planner_spike.sh
```
