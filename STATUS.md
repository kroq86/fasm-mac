# Project status — canonical direction

Updated: 2026-09-02

This file is the single current map of the project.  Historical scratchpads and
spikes preserve evidence, but they do not define the active roadmap.  When a
result changes a verdict or the next gate, update this file in the same change.

## One active direction

The active engineering direction is an inspectable native decoder runtime,
using public GPT-2 124M weights as the first real language-model boundary:

```text
foreign GPT-2 weights
  -> inspect/fingerprint
  -> one real block differential
  -> full 12-block logits differential
  -> self-written tokenizer
  -> autoregressive generation
  -> planner optimization on the verified model
```

The public ONNX Model Zoo `mnist-8.onnx` remains a completed inspect/layout
case study. Its native CNN build is backlog, not a prerequisite for GPT-2.

## Current tensorctl facts

- `inspect/build/verify` passes for the narrow Wine MLP path.
- The current importer reads the public MNIST graph and reports shapes,
  initializers, MAC estimates, NCHW layout boundaries, and the HWC-flatten
  weight-order hazard.
- Rebuilt from the current working tree, MNIST `build` exits nonzero and leaves
  no artifact.  The first reported blockers are missing native lowering for
  `Reshape`, `Conv`, and `MaxPool`, plus two int64 shape initializers rejected
  by build's float32-only initializer preflight.
- Therefore any statement that the present first blocker is “exactly one graph
  input” is obsolete or unverified against the current build.  Re-run after
  each importer/build change; do not carry an earlier diagnostic forward.

## Current decoder-runtime progress

- Operation 1/10, `EMBED_LOOKUP`: differential forward/backward gate passes on
  sequence lengths 7/8/9, including duplicate-token scatter-add.
- Operation 2/10, positional embedding composition: differential gate passes.
- Operation 3/10, causal attention: differential forward, causal isolation,
  row sums, and finite-difference backward gates pass.
- Operation 4/10, GPT-2 tanh-approximation `GELU`: differential gate passes.
- Operation 5/10, stable row softmax: differential forward/backward and
  overflow/underflow gates pass.
- Operation 6/10, top-k/temperature sampling: property/differential and hostile
  contract gates pass. Coverage includes zero-seed normalization, non-finite
  and null inputs, `k >= vocab` full-vocabulary behavior, explicit top-k and
  vocabulary capacity rejection, deterministic replay, statistical agreement,
  ASan bounds checks, and absence of shared mutable selection scratch.
- Operation 7/10, KV cache: not a new canonical op (the canonical
  `CAUSAL_ATTENTION` kernel is fixed-T at compile time, so it cannot itself
  represent a growing context). A runtime-length-parameterized full-recompute
  reference is anchored against the real canonical op+executor at T=7/8/9;
  a hand-rolled incremental append-and-cache implementation is then checked
  against that anchored reference at every context length, plus a
  non-retroactivity check (appending more tokens never changes already-
  produced outputs). All three checks pass at T=7/8/9.
- Operation 8/10, autoregressive loop: integrates every operation above into
  one fixed-weight toy decoder block (embed+pos -> QKV proj -> causal
  attention -> output proj -> FFN with GELU -> logits proj -> softmax ->
  top-k/temperature sample -> append) and one generation loop. The full
  pipeline is anchored against the real canonical graph+executor at T=7/8/9;
  the incremental KV-cache generation path is checked step-by-step against a
  full-recompute reference; replay determinism and prefix-stability (a
  shorter generation is an exact prefix of a longer one from the same seed)
  both pass. Weights are fixed/untrained and the vocabulary/tokenizer are
  toy-scale — this proves the loop's wiring and KV-cache correctness only,
  not anything about a real model's output quality.

The original plan listed 10 conceptual decoder-runtime items; two of them
(LayerNorm, logits projection) were already satisfied by pre-existing
`LAYERNORM`/`MATMUL` ops and never needed their own gate, which is why the
operation log above only numbers 1-8/10 (the 8 items that needed new
kernels or new integration logic: embedding lookup, positional embedding
composition, causal attention, GELU, softmax, sampling, KV cache,
autoregressive loop). All 8 are now gated and passing, so the decoder-runtime
op-matrix milestone is closed. It is still all toy-scale synthetic data and
fixed/untrained weights, not GPT-2 itself.

## Real-block gate: LOCAL EVIDENCE PASS, PORTABILITY OPEN

The numerical one-real-block workflow below is implemented and passes on this
machine against the locally cached, fingerprint-verified artifact:

```text
public GPT-2 124M weights
  -> load one real Transformer block
  -> execute the block through the canonical runtime
  -> compare its hidden state with a reference implementation
```

This was a **runtime correctness boundary**, not a tokenizer or model-quality
demo. Fixed integer token IDs were used throughout; no text tokenization was
involved anywhere in this gate.

`fasm/spikes/tensor_gpt2_block0_differential_check.c`, gated by
`scripts/check_tensor_gpt2_block0_spike.sh`, wired into
`scripts/check_tensor_runtime_spikes.sh`. Model: real `gpt2` (124M, 12-layer,
`n_embd=768 n_head=12 n_inner=3072 vocab=50257`), huggingface.co/gpt2, revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, `model.safetensors`, MIT license,
548105171 bytes, sha256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`
— **not** the locally-present 6-layer DistilGPT2 (`/Users/ll/distilgpt2`),
which was checked and rejected for this purpose. Block 0, `T=8`, fixed
arbitrary token IDs, eval mode (dropout disabled).

Result, preregistered tolerance `abs(diff) < max(1e-2, 1e-3*|ref|)` per
element, checked at every named boundary in causal order (diagnostics stop at
the first boundary exceeding tolerance — none did):

| boundary | elements | max abs err | max rel err | fails |
|---|---|---|---|---|
| `ln_1` (real affine) | 6144 | 1.19e-07 | 1.07e-04 | 0 |
| QKV | 18432 | 8.11e-06 | 9.84e-03 | 0 |
| attention output | 6144 | 1.24e-05 | 5.94e-03 | 0 |
| first residual | 6144 | 1.24e-05 | 9.20e-03 | 0 |
| `ln_2` (real affine) | 6144 | 9.09e-07 | 2.03e-03 | 0 |
| GELU output | 24576 | 1.43e-05 | 15.2 | 0 |
| final hidden state | 6144 | 3.81e-05 | 4.35e-03 | 0 |

(The GELU boundary's large max-relative-error number is an artifact of
near-zero reference values in the denominator, not a real discrepancy — its
max absolute error is 1.43e-05, same order as every other boundary; the pass
condition used the combined absolute-floor-or-relative bound, not relative
error alone.) 0 non-finite values anywhere. Reran twice, byte-identical
native stdout both times. The mismatch-coordinate/stop-at-first-boundary
reporting path was itself verified by forcing an artificial near-zero
tolerance in a throwaway build and confirming it correctly stopped at `ln_1`
and printed the exact worst-mismatch index/values, rather than being trusted
unexercised.

**Self-written runtime boundary — what the local run proves:**
- Weights are read directly from the real `model.safetensors` file by a new
  self-written safetensors parser (`fasm/spikes/tensor_safetensors_loader.h`,
  gated by `check_tensor_safetensors_loader_spike.sh`, its own tensor reads
  checked against an independent raw-struct/json Python oracle) — no
  hand-dumped/reshuffled `.f32` weight copies. An earlier draft of this gate
  did dump weights to flat `.f32` files first; that path was identified as
  violating this exact rule and was replaced before being counted as done.
- The artifact fingerprint is verified by a new self-written SHA-256
  (`fasm/spikes/tensor_sha256.h`, gated by `check_tensor_sha256_spike.sh`,
  checked against Python's `hashlib` on empty/short/block-boundary/
  multi-block messages) computed over the real 548MB file at gate run time,
  not merely asserted in a comment.
- `torch`+`transformers` were used **only** as an isolated, disposable test
  oracle (`scratchpad/gpt2_block_boundary/generate_reference.py`, its own
  venv under `scratchpad/gpt2_block_boundary/venv311/`, never committed,
  never a build/runtime dependency) to produce the reference input hidden
  state and every named intermediate boundary tensor. It does **not** dump
  weights.
- LayerNorm gap: this project's `LAYERNORM` kernel has no learnable affine.
  Rather than add a new op, real GPT-2's `gamma`/`beta` are folded into the
  immediately following `MATMUL`+`BIAS_ADD` at load time — an exact
  linear-algebra identity (`(LN_raw(x)*gamma+beta)@W+b == LN_raw(x)@(diag(gamma)@W)+(beta@W+b)`),
  verified numerically offline (max abs diff 8.1e-6, pure float rounding)
  before being trusted, and independently re-confirmed by the `ln_1`/`ln_2`
  boundary comparisons above (which recompute the real post-affine value
  outside the graph purely to diff against the reference).

**Known gap against the acceptance contract below:** run-twice determinism is
satisfied, but the gate's default artifact path is hardcoded to this
machine's local Hugging Face cache directory, not a documented portable
fetch-helper. A clean checkout elsewhere needs the file at the recorded
revision/sha256 placed manually or passed via argv; there is no automated
fetch step yet. In addition, the current check scripts return success with a
`skipped` message when the model or reference fixtures are missing. That is
acceptable for the broad optional regression suite, but cannot satisfy this
milestone's required clean-checkout gate. The portable gate needs a required
mode in which missing artifacts/fixtures are nonzero failures.

### Self-written runtime boundary

- The production path must remain implemented in this repository's native
  C/FASM compiler/runtime. It must not link to or invoke PyTorch,
  `transformers`, ONNX Runtime, Python, or another ML framework.
- A third-party framework may be used only as an isolated **test oracle** to
  produce reference tensors. It is not a build/runtime dependency and must not
  be vendored into the repository or installed into a repository-owned virtual
  environment.
- Prefer already available tooling. Any oracle environment must be disposable
  and external to the repository. The committed evidence consists of small
  fingerprinted fixtures plus the exact generator/version/source recipe, not a
  copied Python environment.
- The native path must read the public weight artifact automatically. No manual
  transposition, renaming, slicing, or rewriting of model weights is allowed.
  Layout conversion is acceptable only when performed explicitly by the
  loader/lowering code and reported in its trace.

### Real-block acceptance contract

Before implementation, record the exact model identity, download/source URL,
immutable revision, license/provenance, file format, byte size and SHA-256.
Network fetching must be an explicit helper; the gate must also accept an
offline local-artifact path. `GPT-2 124M` here means the 12-layer GPT-2 small
configuration (`n_embd=768`, `n_head=12`, `n_inner=3072`, `vocab=50257`), not
the locally present six-layer DistilGPT2 model. The gate must then prove all of
the following on block 0 at fixed `T=8` in evaluation mode (dropout disabled):

1. Inspect and validate model identity/configuration, artifact fingerprint,
   tensor inventory, required names, shapes, dtypes, parameter bytes and every
   layout conversion before execution; fail closed on missing or incompatible
   data. This inspection is part of the gate, not a manually prepared note.
2. Map the real `ln_1`, packed QKV projection and bias, attention output
   projection and bias, `ln_2`, MLP expansion/projection weights and biases
   without model-specific hand editing.
3. Implement GPT-2's actual pre-LayerNorm residual order, causal mask,
   multi-head split/merge, weight orientation, LayerNorm epsilon, and
   tanh-approximation GELU semantics. Do not substitute the earlier toy block's
   post-LayerNorm/no-bias structure.
4. Use one fixed, fingerprinted input and compare named intermediate boundaries
   (`ln_1`, QKV, attention output, first residual, `ln_2`, MLP GELU/output) as
   well as the final block hidden state. Stop diagnostics at the first boundary
   exceeding tolerance.
5. Pre-register absolute/relative tolerances before seeing the native result;
   report maximum absolute error, maximum relative error, mismatch coordinates,
   and NaN/Inf status. Raw-byte equality may be reported diagnostically but is
   not the semantic pass condition.
6. Run twice and reproduce identical native output bytes on the same machine.
   The gate must run from a clean checkout using a documented artifact-fetch or
   local-artifact path and must leave no generated dependency environment in
   the repository.

Current local caveat: `/Users/ll/distilgpt2` is DistilGPT2 (`n_layer=6`); it
was checked and correctly NOT used as a stand-in for the declared GPT-2 124M
target above. A disposable Python venv (`torch`+`transformers`, currently
~865MB on disk)
exists under `scratchpad/gpt2_block_boundary/venv311/`, used only as the
isolated test-oracle instrument this section requires; it produces small
reference tensors (~200KB total: input hidden state + 6 named boundaries +
final output) plus `manifest.json`/`gate_config.txt`. Neither the venv nor
the fixtures are committed. Its location inside the repository violates the
artifact-hygiene rule above even though it is not a runtime dependency.
Preserve it until the owner explicitly chooses cleanup, but a portable oracle
recipe must create any environment outside the checkout.

The real-block differential above has passed locally (see "Real-block gate:
LOCAL EVIDENCE PASS, PORTABILITY OPEN").
Do not start tokenizer, full 12-layer generation, KV-cache optimization, new
ONNX operators, or SIMD tuning as a result of that pass alone — the ordered
roadmap below still gates each of those behind the preceding step, and the
fetch-helper gap noted above should be closed before treating this milestone
as fully portable.

## Ordered roadmap after the portable real-block gate

First close the remaining portability work: add the pinned fetch/offline-input
helper, generate the oracle environment outside the checkout, fingerprint the
small reference fixtures, and run a required (non-skipping) clean-checkout
gate. Each item below starts only after that and the preceding gate pass:

1. **Full-logits correctness:** execute all 12 real GPT-2 blocks, final
   LayerNorm and the tied token-embedding/LM head from fixed token IDs; compare
   final hidden states and raw logits with the independent oracle. This step
   still requires no tokenizer.
2. **Self-written tokenizer:** implement GPT-2 byte-level BPE in the native
   product path and verify token IDs and byte round-trips against a fixed corpus
   containing ASCII, whitespace, Unicode and byte-fallback cases. External
   tokenizer libraries remain test oracles only.
3. **Generation correctness:** connect tokenizer, full logits and the already
   gated sampling contract; verify greedy generation exactly and stochastic
   generation by deterministic seed replay plus distribution/property tests.
4. **Canonical KV-cache integration:** replace the hand-written toy cache path
   with planner-visible runtime-length state, then compare every generated-step
   logit against full recomputation before measuring speed.
5. **Planner as a virtual CGRA experiment:** on the now-correct real model,
   compare fixed lowering with planner-selected CPU kernels, layouts, copies,
   fusion, save/rematerialize and memory placement. The preregistered question
   is whether the planner improves TTFT, tokens/s or peak memory without
   violating numerical equivalence. A synthetic graph is not sufficient
   evidence for this claim.

No miniature GPT detour is required between the operation matrix and the real
block: it would repeat the toy boundary without answering the next unknown.

## Frozen research branch: neuro-symbolic / SparseOmega

This branch is retained as evidence but is not the active roadmap.

| Experiment | Final status | Defensible conclusion |
|---|---|---|
| Full-state evolution replay | `SUPPORTED` | Complete recorded state reproduces deterministic evolution. |
| Transformer -> typed proposal -> validated executor | `SUPPORTED` as implementation | Typed lazy `IF`, arithmetic, transactional state, fail-closed validation, and replay work in the toy system. This is not novel LLM reasoning. |
| First atomic composition experiment | `INVALID` | It requested unseen atomic class labels and did not test compositionality. |
| Factorized held-combination experiment | `NEGATIVE` | Overall whole-bundle composition claim failed; a positive hard-stratum result does not override the preregistered primary endpoint. |
| Exact causal traversal vs relaxation | `NEGATIVE / NO_ADVANTAGE` | Relaxation approximated a task solved exactly by ordinary graph traversal. |
| Weighted contradictory relations | `NO_ADVANTAGE_OVER_CLASSICAL` | Relaxation was selective, but an exact solver at matched 0.35 coverage had higher accepted accuracy (`0.9196` vs `0.9018/0.9048`). |
| Hebbian sequential/adaptation branch | `INCONCLUSIVE_INSTRUMENT_SENSITIVITY` | The final period/readout was too short for the graph time constants. Three-attempt branch closed; do not retune it post hoc. |
| Embedded vs external executor | narrow engineering result only | Warm persistent IPC was much slower for tiny trusted typed operations in one Rosetta environment. This is an expected systems trade-off, not evidence that general LLM tool use is wrong. Multi-run CI and incremental host footprint remain unclosed. |

Do not create another neuro-symbolic, relaxation, Hebbian, or embedded-vs-tool
spike unless a new unknown claim, closest prior art, matched baseline,
falsifier, and decision-changing outcome are written down first.  The current
decision is to stop this branch.

## Paper status

There is no supported “revolution in LLM reasoning” paper claim.  A future
paper, if earned, is about an inspectable native ML runtime and the evidence it
makes observable: foreign-model import/lowering/verification, execution-plan
provenance, memory/liveness decisions, and differential numerical behavior.
The next evidence boundary for that thesis is the real GPT-2 block, not more
MNIST/CNN coverage. Do not draft or advertise novelty until the real-block gate
passes and a mechanism-level prior-art audit finds a defensible contribution.

## Completion definition

The immediate milestone is complete when a clean checkout can obtain the
pinned public GPT-2 weights, reproduce their fingerprint, execute one real
block, and match the reference hidden state within a preregistered tolerance.
Current project status: **decoder-runtime op-matrix complete (8/8 new
primitives, toy-scale/fixed-weight only); real GPT-2 124M block 0 executes
through the canonical runtime and matches the PyTorch reference at every
named boundary within preregistered tolerance (see "Real-block gate:
RESULT"); this has been verified on this machine with a locally-cached
artifact, not yet from a clean checkout with an automated fetch step — that
gap is the only thing separating "passes here" from "complete" per this
section's own definition**.
