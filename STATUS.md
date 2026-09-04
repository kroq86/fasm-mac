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

The user has explicitly authorized a gated second direction: test whether the
verified runtime can support a state-transition agent rather than becoming
another model runner. `scratchpad/verified_agent_preregistration.md` freezes a
four-step spike ladder (instrument sensitivity, proposer feasibility, matched
three-arm comparison, held composition). No product integration or expanded
world model is allowed before the preceding spike passes.

Verified-agent Spike 1 (instrument sensitivity) now passes via
`scripts/check_verified_agent_spikes.sh`: valid transitions commit atomically;
late invariant failures and stale revisions leave state unchanged; replay uses
a canonical little-endian byte capsule rather than C-struct layout; format
version, executor fingerprint and checksum are checked; corrupted/unknown-
version capsules fail closed; and disabling verification produces the expected
observable invariant breach. This validates the measurement instrument only,
not agent usefulness. Spike 2 (language-proposer feasibility) is the next gate.

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

## Real-block gate: COMPLETE

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

Portability is closed by `scripts/fetch-gpt2-124m.sh` plus the explicit
`--model` offline path and `--required` gate mode. The fetch helper pins the
revision, byte size and SHA-256 and publishes atomically only after validation;
required mode treats missing model/fixtures as failures. The eight small block-0
reference tensors are committed with per-file SHA-256 fingerprints.

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
target above. The one-off Python oracle environment was moved outside the
checkout to `/Users/ll/.cache/fasm-mac-oracles/gpt2-block0-venv311`; it remains
test-only and is not a runtime dependency. The eight block-0 reference tensors,
their fingerprints, `manifest.json`, and `gate_config.txt` are committed so the
required differential does not need to regenerate the oracle.

The real-block differential above has passed, including a detached clean
worktree run of commit `b743910` using the explicit offline artifact path; the
worktree remained clean after the required gate (see "Real-block gate:
COMPLETE").
Do not start tokenizer, KV-cache optimization, new ONNX operators, or SIMD
tuning as a result of that pass alone — the ordered roadmap below still gates
each of those behind the preceding step.

## Full-logits gate (roadmap item 1): COMPLETE

`fasm/spikes/tensor_gpt2_full_differential_check.c`, gated by
`scripts/check_tensor_gpt2_full_spike.sh` (same `--required`/`--model`/
`--fixtures` interface as the block-0 gate), wired into
`scripts/check_tensor_runtime_spikes.sh`. Same model/fingerprint/provenance as
the block-0 gate above (real `gpt2`, revision `607a30d7...`, sha256
`248dfc39...`). All 12 real blocks, `ln_f`, and the tied LM head (`wte.weight`
reused as the output projection, per real GPT-2 — explicitly transposed once
at load time from its `[VOCAB,M]` embedding-table orientation into `[M,VOCAB]`
for this project's `MATMUL` convention, a reported layout conversion with no
value edit). Still no tokenizer, no generation, no training — fixed integer
token IDs throughout, same as block-0.

The canonical compiler caps a single graph at `MAX_NODES=64` (a shared
constant, left untouched). A full 12-block+`ln_f`+logits graph needs ~290
nodes, so this runs as a sequence of small per-stage `compile()`+`execute()`
calls (embedding, then one call per block, then `ln_f`+logits), each well
under the cap, threaded through plain float buffers — no gradient/backward
needed here, so this changes nothing about correctness, the same pattern the
KV-cache work already used for its own fixed-graph-size constraint.

Result, preregistered tolerance `abs(diff) < max(3e-2, 2e-3*|ref|)` (widened
from the single-block gate's `max(1e-2,1e-3*|ref|)` to account for 12x the
sequential accumulation, chosen before running), boundaries checked in causal
order, diagnostics stop at the first exceeding tolerance — none did:

| boundary | elements | max abs err | max rel err | fails |
|---|---|---|---|---|
| hidden state after block 5 | 6144 | 7.32e-04 | 1.73e-03 | 0 |
| hidden state after block 11 (last) | 6144 | 6.71e-04 | 0.102 | 0 |
| `ln_f` (real affine) | 6144 | 5.34e-05 | 3.96e-03 | 0 |
| final logits | 402056 | 2.90e-04 | 3.34e-06 | 0 |

(`layer11`'s max-relative-error is again a near-zero-reference-value artifact,
not a real discrepancy — its absolute error is the same order as every other
boundary.) 0 non-finite values anywhere. Reran twice, byte-identical native
stdout both times. **The greedy top-5 predicted tokens at the last position
exactly match the PyTorch oracle** — `[471, 717, 968, 1708, 1578]`, logits
matching to 4 decimal places (`-64.5787, -64.7665, -64.8301, -64.8751/2,
-65.0124`) — the concrete falsifier this whole gate was named around
("unlocalizable accumulated numerical error changing the greedy sequence")
did not trigger.

One real bug this check caught in itself before it passed: `ln_f`'s raw
(pre-affine) `LAYERNORM` output was initially compared directly against the
Python oracle's real post-affine `ln_f` output — the same mistake this
project's `ln_1`/`ln_2` fold already had a fix for in the block-0 gate, just
not re-applied here. Caught by an 193-magnitude boundary mismatch, fixed by
recomputing the real post-affine value the same way block-0 does, purely for
the diff (the actual logits computation already used the correct fold).

Fixture fingerprint manifest (`reference_sha256.txt`) covers both the block-0
and full-model reference tensors together in the same directory; regenerated
once already after the parallel provenance-metadata upgrade to `gate_config.txt`/
`manifest.json` changed their bytes (fixture *values* unchanged, `shasum -c`
now green again). The full-model fixture generator
(`scratchpad/gpt2_block_boundary/generate_full_reference.py`) still writes
inside the checkout, unlike the newer `generate_reference.py`'s
`GPT2_ORACLE_OUT`-outside-checkout convention — bringing it in line with that
convention is a reasonable follow-up, not yet done.

## Self-written tokenizer (roadmap item 2): COMPLETE (ASCII scope)

`fasm/spikes/tensor_gpt2_bpe.h` (loader/encode/decode), differential/property
check `fasm/spikes/tensor_gpt2_bpe_differential_check.c` + embedded oracle
test cases `tensor_gpt2_bpe_test_cases.h`, gated by
`scripts/check_tensor_gpt2_bpe_spike.sh`, wired into
`scripts/check_tensor_runtime_spikes.sh`. Real, pinned `vocab.json`
(1042301 bytes, sha256 `19613966...`) / `merges.txt` (456318 bytes, sha256
`1ce16647...`), huggingface.co/gpt2, revision `607a30d7...` (same repo as
the model weights), fetched by the new `scripts/fetch-gpt2-tokenizer.sh`
(same pinned-size/SHA-256/atomic-publish pattern as `fetch-gpt2-124m.sh`).

**Scope decision, made explicitly with the user before implementation:**
GPT-2's real pre-tokenization regex classifies letters/digits via Unicode
`\p{L}`/`\p{N}` categories; this project has no Unicode category tables, so
that classification is ASCII-only here. The byte<->codepoint table itself
(the other half of real byte-level BPE) is the complete, real algorithm —
only pre-tokenization's classification of the input text is scoped down.
Round-trip correctness is therefore claimed for ASCII input only; non-ASCII
bytes still reach BPE merging (falling into the "other" class rather than
being letter/number-classified) and are required not to crash, not claimed
correct.

21 embedded test cases (contractions, multi-space/tab/newline runs,
punctuation, digits, case mixing, empty/whitespace-only/single-char
boundaries, vocab id 0 and id 50256) generated from an independent
from-scratch Python oracle run against the same pinned files — 21/21 match
on encode ids, and ASCII round-trip (encode→decode) is exact on every
nonempty case. Additional property checks: non-ASCII input handled without
crash (scope not claimed), vocab-boundary ids (0, 50256 = the real
`<|endoftext|>` piece) decode correctly, out-of-range token id fails closed
on decode. Reran twice, byte-identical native output both times.

**Two real bugs this check caught before it passed** (full detail in the
differential-check file's header comment):
1. The BPE merge loop merged only the first occurrence of the winning-rank
   pair per iteration; the reference algorithm merges every non-overlapping
   occurrence in one pass — these can diverge. Fixed to match.
2. The independent Python oracle itself had a bug: its merges.txt parser
   treated any line whose first piece is the literal `#` character as a
   comment (correct only for line 0), silently dropping 8 real merge rules
   (including the legitimate rule `# $`) and producing wrong "expected"
   values for cases exercising them. Found by checking a failing case's
   ranks against the raw file directly rather than assuming the new C code
   was at fault; fixed by only slicing off the header line, matching
   OpenAI's reference `encoder.py` exactly.

Known, explicit non-goals per the roadmap: no special-token handling
(`"<|endoftext|>"` literal text encodes as ordinary BPE pieces, not id
50256 — both the oracle and this implementation agree on that, it is not a
claim about special-token correctness); no generation loop yet (item 3).

## Generation correctness (roadmap item 3): COMPLETE

`fasm/spikes/tensor_gpt2_forward.h` (new: a runtime-sequence-length GPT-2
forward pass — this project's canonical compiler bakes every shape,
including `T`, into compile-time macros, so one compiled graph only ever
executes at one fixed length; a generation loop needs a growing length at
every step). Mirrors the canonical graph's exact per-op formulas as plain
runtime-`n`-parameterized C, the same "hand-rolled runtime reference,
anchored once against the real canonical op at a fixed shape" pattern
already used for the toy KV-cache/autoregressive-loop checks — now with
the real GPT-2 124M weights and all 12 real blocks. Differential check
`fasm/spikes/tensor_gpt2_generation_differential_check.c`, gated by
`scripts/check_tensor_gpt2_generation_spike.sh`, wired into
`scripts/check_tensor_runtime_spikes.sh` (~2-2.5min per run — a real 124M
forward pass at growing context, unoptimized scalar C, no KV-cache yet by
design; performance is item 5+, not this gate).

**Anchor:** `gpt2_forward_ex`'s logits at the same fixed `n=8` input this
project already ran through the real canonical `compile()`+executor
(`tensor_gpt2_full_differential_check.c`) are compared against that same
PyTorch reference fixture — 402056 elements, max abs err 2.98e-04, max rel
err 3.41e-06, 0 fails against the same preregistered tolerance as the
full-logits gate. This transitively anchors the runtime-`n` forward pass to
the already-verified canonical/executor path: same weights, same math,
same external reference. This proves single-step logits agreement at one
fixed length only — it does NOT by itself prove the generation loop (repeated
forward + argmax + append) tracks the real model over many steps, which is
why item 1b below exists as a separate, stronger check (added after review:
the anchor plus internal self-consistency was correctly flagged as
insufficient to claim "our greedy generation matches PyTorch's").

**Token-for-token greedy cross-check (the actual generation-loop claim,
not just single-step logits):** real PyTorch greedy sequences for 4 prompts
(`scratchpad/gpt2_block_boundary/greedy_reference.txt`, generated by
`generate_greedy_reference.py` in the same directory, torch==2.13.0/
transformers==5.16.1 in a disposable external venv) — 12 generated tokens
each, compared one by one against this project's own greedy generation from
the same prompts. **48/48 tokens match exactly across all 4 prompts**,
including a deliberately out-of-distribution prompt (`[1,2,3,4,5]`) where
real GPT-2's own greedy decoding degenerates into a repeating `2,5,2,5,...`
loop — this project's runtime reproduces that exact degenerate repetition,
not just "plausible-looking" output. (One prompt, `[464,3290,3332,2159]`,
was independently cross-checked by the user running PyTorch by hand outside
this gate, with a matching result, before this formal embedded check
existed.) Environment note: a length-1 initial sequence reliably crashes
this machine's torch/transformers build with SIGBUS (reproduced directly,
no Python traceback) — an oracle-environment issue, not a bug in this
project's code; no length-1 prompt is used, shortest tested is 2 tokens.

**Properties, all real GPT-2 124M, no toy weights:**
- Greedy generation: two independent runs from the same prompt produce a
  byte-identical 10-token sequence.
- Temperature/top-k generation (using the already-gated
  `sample_top_k_temperature` contract): same seed reproduces an identical
  10-token sequence; an 8-token generation from the same seed is an exact
  prefix of the 10-token one.
- Real end-to-end demo (encode → generate → decode, this project's own
  tokenizer both ends): prompt `"The quick brown fox"`, 12 tokens greedy
  continuation → `"The quick brown foxes are a great way to get a little
  bit of a"` — legible, grammatical English, produced by the real GPT-2
  124M weights running entirely through this project's self-written
  runtime (canonical-graph-equivalent math, safetensors loader, BPE
  tokenizer). Not itself a pass/fail gate (no "good continuation" oracle
  exists) but required to encode/generate/decode without error.

Not done here, explicitly deferred to later roadmap items: KV-cache (item
4 — every step above is a full recompute from scratch); performance
measurement (item 5+); backward/training (items 6-8); the planner
experiment (item 9); GPU (item 10).

## Canonical KV-cache integration (roadmap item 4): CORE CLAIM COMPLETE

A new canonical op, `CAUSAL_ATTENTION_CACHED`, was added to
`tensor_semantic_compiler.h` (design confirmed with the user before
implementation: a new op taking one new token's Q/K/V — T=1 — plus an
externally-owned cache tensor, rather than trying to make `T` itself
runtime-variable, which the shared compiler's fixed-shape-macro
architecture cannot support without a much larger rewrite). Full doc
comment in that file explains the design; two real bugs were caught and
fixed while landing it (both documented there and in the differential
check's header): validate()'s blanket "every tensor has rows≥1" rule
rejects a brand-new empty cache, fixed by making `rows` the fixed capacity
and `aux_count` the mutable position counter instead; and the cache leaf
was first flagged `PARAM`, colliding with an unrelated existing convention
(`PARAM` + non-null `aux` means optimizer metadata) — fixed by using
`INPUT` (this cache is inference-only state, never an optimizer target).

**Isolated verification** (toy data, `fasm/spikes/
tensor_causal_attention_cached_differential_check.c`, gated by
`scripts/check_tensor_causal_attention_cached_spike.sh`): three-way
cross-check at T=7/8/9 — the new canonical op vs. this project's own
already-anchored runtime-n full-recompute reference vs. the earlier toy
KV-cache spike's hand-rolled incremental function — exact agreement at
every step, cache-state (`aux_count`) advances correctly as the kernel's
one documented side effect, non-retroactivity holds, and a full cache is
correctly rejected by `validate()` (fail-closed, exercised directly).

**Real-model integration** (`fasm/spikes/
tensor_gpt2_cached_generation_differential_check.c`, gated by
`scripts/check_tensor_gpt2_cached_generation_spike.sh`): real GPT-2 124M,
same fingerprinted checkpoint as every other gate. Scope, deliberately
narrow: only the attention step of each decode step needs cache state, so
only that step is a real `compile()`+executor call against the new op;
prefill and the surrounding MATMUL/BIAS_ADD/RESIDUAL/GELU stay the
already-anchored plain runtime-n C (no caching concern for stateless
per-position ops). Result: for 3 prompts, 7 cached-decode steps each — 21
steps total — after a refactor to a persistent-graph pattern (`compile()`
once per layer, execute many times, reusing the same `Node` array across
steps instead of rebuilding it per token), the cached path's logits agreed
with full recomputation **bit-exactly (worst_abs=0)**, not just within the
originally-preregistered 1e-3 tolerance, and produced byte-identical
greedy tokens at every single step, not just a final spot check. One
prompt's cached-decode output was independently cross-checked
against the real PyTorch greedy sequence (`greedy_reference.txt`) too:
exact token match, 12 tokens. Runs in ~30-36s (faster than the
full-recompute generation gate's ~2-2.5min for a comparable prompt/length,
but this is NOT a performance measurement — no controlled/matched timing
methodology, no repeated runs, no isolation from other system load; item 5
is where performance actually gets measured, not this gate).

**What this proves and what it doesn't, precisely:** the cache is real
canonical `Node`/`Tensor` state — its data buffers (`data`=K, `aux`=V) are
genuine Tensor fields the kernel reads and mutates during each
`compile()`+executor call, not synthetic bookkeeping bolted on afterward.
It is **available to a future planner, not yet "planner-visible" in any
operative sense** — no planner pass exists yet that inspects, chooses
layout for, or schedules this op; that's item 9's job, not done here. The
KV **position** counter IS now continuously resident in a persistent
Tensor: `compile()` runs once per layer at setup, and every subsequent
step reuses that same `Node` array and calls the executor directly — the
cache's `aux_count` field is mutated in place by the kernel itself, with
no external C struct copying position in or out between calls. (An
earlier version of this integration did copy position via an external
`LayerCache.pos` struct field, rebuilding the `Node` array every call;
that was flagged as architecturally weaker and replaced by this
persistent-graph pattern.) It does NOT mean the planner can choose
anything about this cache's
layout/placement/scheduling; it does NOT establish speed (item 5); it does
NOT mean generation as a whole "became canonical" — only the cached-attention
step per layer runs through `compile()`+executor, everything else in the
block (MATMUL/BIAS_ADD/RESIDUAL/GELU, the LM head, prefill) remains
hand-written C, unchanged from before this integration. Backward is
deliberately unset for this op (inference-only, item 6 is a separate
undertaking). Prefill still uses full recompute rather than its own
cache-building op — populating
the cache from prefill's already-computed K/V is a data copy, not a new
computation, so this was judged not to need its own canonical op.

## Performance measurement (roadmap item 5): ALGORITHM EXPERIMENT + SUSTAINED GENERATION DONE, MEMORY PARTIAL, CROSS-RUNTIME BLOCKED

Full harness, raw data, provenance: `scratchpad/gpt2_perf_bench/`
(`bench_algorithm.c`, `raw_algorithm.tsv`, `summary_algorithm.txt`,
`PROVENANCE.txt`, `README.md`). Real GPT-2 124M, same fingerprinted
checkpoint as every other gate. Warmup=3 discarded, 10 measured
repetitions per point, mode order alternated each repetition, every
single sample's token AND logits max-abs-diff recorded and checked
against a 1e-3 bound (matching the block-0 gate's own tolerance) before
being kept — a mismatch aborts the run (did not happen; observed max abs
diff was exactly 0 across all samples in the most recent run).
Correctness here is a **self-consistency check** between this project's
own two paths, not an independent oracle run inside this program — both
paths were already independently anchored to real PyTorch by separate,
earlier gates, which is what makes self-consistency an adequate bar for
a *performance* harness specifically.

**Algorithm experiment (VALID — same binary, same ISA):** fasm-mac
full-recompute vs. fasm-mac canonical-KV-cache, marginal cost of one
decode step at context lengths 4/16/32/64 (most recent run; numbers vary
run to run on a shared dev machine, the qualitative pattern is what's
load-bearing):

| ctxlen | cached median | full median | ratio |
|---|---|---|---|
| 4 | 164ms | 497ms | 3.0x |
| 16 | 297ms | 2004ms | 6.8x |
| 32 | 327ms | 3961ms | 12.1x |
| 64 | 280ms | 7457ms | 26.6x |

**Complexity, precisely (corrected after review — an earlier version of
this section wrongly said cached decode is O(1)):** the cached step is
**O(L)**, not O(1) — the new token's query still attends over all L
cached K/V pairs; only the surrounding per-token projections are O(1) in
L. Full recompute is **O(L²)** (each of L positions attends over up to L
others) plus O(L) projections. The measured cached times above are
visibly not flat across L, consistent with O(L). What this experiment
supports: cached decode grows much more slowly with context than full
recompute (O(L) vs O(L²)) — expected, now measured, not a discovery, and
not "constant time".

**Sustained end-to-end generation (`bench_sustained.c`):** one continuous
prefill+decode run (prompt_len=4, gen_len=40), not isolated marginal
steps. A real bug was found and fixed here (see `README.md` §1b): the
first version double-wrote a cache entry for the last prompt token,
corrupting the cache from step 0 and producing an unnoticed-until-step-24
argmax flip; root-caused by comparing the live incremental path against
fresh full recompute at every step, not just at the sequence's end. After
the fix, bit-exact (`max_abs=0`) agreement at all 40 steps. Result (single
run, not yet warmup/repeated like the algorithm experiment above):

| mode | ttft_ms | decode tokens | tokens/sec |
|---|---|---|---|
| cached | 590.0 | 39 | 6.47 |
| full | 508.0 | 39 | 0.33 |

~19.4x sustained throughput ratio, consistent in direction with the
algorithm experiment. TTFT is comparable here only because the prompt is
short; not yet measured at longer prompts.

**Still not measured, so item 5 is not fully closed:** mode-attributable
memory beyond KV-cache capacity (the recorded `process_peak_rss_kb` is a
whole-process high-water mark shared by both modes in one process, not a
per-mode number; the one genuinely mode-specific memory figure is the
KV-cache capacity itself, ~9MB total, reported in the harness's
PROVENANCE line); and repeated/warmup-controlled sustained-generation
runs (currently a single sample).

**Implementation experiment (fasm-mac vs. llm.c): BLOCKED,
`CROSS_RUNTIME_INCONCLUSIVE`.** fasm-mac's canonical executor is x86_64
FASM only (`tensor_transformer_executor_f32.asm`); this machine is Apple
M1, so fasm-mac runs under Rosetta while llm.c runs native arm64. No
timing methodology corrects for an ISA mismatch. Earlier informal
single-run numbers comparing the two (no warmup, no repeats, mixed ISA)
were reported before this methodology existed and are superseded — not
evidence, must not be cited. Two possible unblock paths, neither
attempted yet: a native arm64 implementation of the canonical executor's
`ExecStep` ABI (a separate, not-yet-scoped port), or — likely the smaller
lift — building llm.c itself under x86_64/Rosetta so both sides share
fasm-mac's current ISA instead of porting fasm-mac.

**Accelerate SGEMM spike and CLI integration (`bench_accelerate_sgemm.c`,
`bench_accelerate_e2e.c`):** per-op profiling showed >99% of wall time in
matmul-shaped ops (`profile_ops.txt`). Isolated-kernel swap (scalar loop
vs. `cblas_sgemm`, real weights, verified output, 1e-3 tolerance,
warmup+10 alternating reps): 6-136x speedup depending on shape,
independently reproduced. Wired into the real prefill+cached-decode path
(same pattern as `tensorctl gpt2`, not a modification of it): generated
token sequences byte-identical to the scalar baseline across repeated
runs, sustained decode throughput 6-10x faster end-to-end (lower than the
isolated-kernel number, as expected — Amdahl's law, real decode steps
spend time outside the swapped matmuls too). `tensorctl gpt2` now exposes
`--backend accelerate|scalar`, defaults to Accelerate on macOS, and retains
the scalar fallback; its product gate requires identical generated text from
both backends. A formal warmup+10-rep E2E benchmark remains open, so the exact
multiplier is engineering evidence, not a general performance claim. No
planner, Metal/MPS, or general runtime-superiority claim is made here.
The CLI emits runtime metadata to stderr on every generation: selected backend,
model identity and SHA-256, model dimensions, prompt/generated/context token
counts, model/tokenizer load time, prefill, TTFT, post-first-token decode time
and throughput, total inference time, peak RSS, sampling policy and KV-cache
kind/capacity. Generated text remains the only stdout payload.
Generation supports deterministic top-k/temperature sampling through the
already-gated sampling utility (`--temperature`, `--top-k`, `--seed`), while
temperature zero preserves greedy argmax as the default. The product gate
checks same-seed replay and rejects non-finite temperature and unsupported
top-k workspace sizes.

The first real GPT-2 kernel-selection experiment is available as
`tensorctl gpt2 --backend auto`. It profiles scalar and Accelerate candidates
in alternating order (three samples each) for every distinct real matmul shape,
rejects non-finite output or max absolute disagreement above 1e-3, records both
medians and the selected candidate, and reports profiling cost. On the initial
M1/Rosetta run (prompt length 4), it selected Accelerate for all nine distinct
prefill/decode shapes; profiling cost about 275 ms and did not discover a
different regime. This is a negative result for the current two-candidate
planner, not evidence of planner value. Accelerate therefore remains the
production default and `auto` remains an explicit diagnostic experiment.

## Ordered roadmap after the real-block gate

The portability work is complete. Each item below starts only after the
preceding gate passes:

1. **Full-logits correctness — DONE, see "Full-logits gate (roadmap item 1):
   COMPLETE" above.** Executed all 12 real GPT-2 blocks, final LayerNorm and
   the tied token-embedding/LM head from fixed token IDs; final hidden states
   and raw logits matched the independent oracle within preregistered
   tolerance, including exact greedy top-5 agreement. No tokenizer was used.
2. **Self-written tokenizer — DONE (ASCII scope), see "Self-written tokenizer
   (roadmap item 2): COMPLETE (ASCII scope)" above.** Implemented GPT-2
   byte-level BPE in the native product path; verified token IDs and ASCII
   byte round-trips against 21 cases from an independent oracle, all
   matching, plus fail-closed/boundary property tests. Full Unicode
   `\p{L}`/`\p{N}` pre-tokenization classification was explicitly descoped
   (no Unicode category tables in this project) — an open item if ever
   needed, not silently claimed done.
3. **Generation correctness — DONE, see "Generation correctness (roadmap item
   3): COMPLETE" above.** Connected the tokenizer, full logits and the
   already-gated sampling contract; greedy generation verified exactly
   token-for-token against real PyTorch on 4 prompts (48/48 tokens),
   stochastic generation verified by deterministic seed replay plus
   prefix-stability. **Important boundary of this result:** the generation
   path (`tensor_gpt2_forward.h`) is a hand-written runtime-length C forward
   pass, not the canonical `compile()`/executor/planner path (which cannot
   represent a growing sequence length at all — every shape is a
   compile-time macro). Its correctness is proven by anchoring its logits
   to the same real PyTorch reference the canonical path was itself already
   verified against, not by literally running through the canonical
   executor at every generation step. Item 4 below is what actually brings
   generation under the canonical/planner path.
4. **Canonical KV-cache integration — CORE CLAIM DONE, see "Canonical
   KV-cache integration (roadmap item 4): CORE CLAIM COMPLETE" above.** A
   new canonical op (`CAUSAL_ATTENTION_CACHED`) makes the cache real
   `Node`/`Tensor` state, available to a future planner (no planner
   analyzes it yet); real GPT-2 124M decode through real
   `compile()`+executor calls against it agrees with full recomputation
   within 1e-3 (not bit-exact) at every generated step (21/21 steps across
   3 prompts) with byte-identical greedy tokens, and matches real PyTorch
   greedy generation. Speed was NOT measured (that's item 5, and no
   controlled methodology exists yet even informally). Only the
   cached-attention step per layer runs through `compile()`+executor —
   MATMUL/BIAS_ADD/RESIDUAL/GELU, the LM head, and prefill remain
   hand-written runtime-length `tensor_gpt2_forward.h`
   kernels (deliberately narrow scope — those ops have no caching concern
   of their own); bringing them under canonical `compile()` calls too,
   if ever wanted, is a separate, not-yet-justified undertaking.
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
The next evidence boundary for that thesis is full GPT-2 logits, not more
MNIST/CNN coverage. Do not draft or advertise novelty until the full-logits
gate passes and a mechanism-level prior-art audit finds a defensible
contribution.

## Completion definition

The Stage-3 milestone is complete: a clean detached worktree can consume the
pinned GPT-2 artifact through the explicit offline path (or obtain it with the
pinned fetch helper), reproduce its fingerprint, execute block 0, and match all
reference boundaries within the preregistered tolerance.

Stage-4 (all 12 blocks + final LayerNorm + tied LM-head logits from fixed
token IDs) also passes locally now — see "Full-logits gate (roadmap item 1):
COMPLETE" above — including exact greedy-token agreement with the oracle, not
just small numerical error. It has not yet been proven from a clean detached
worktree the way Stage-3 was (the full-model fixture generator still writes
inside the checkout); that portability step is the honest gap before calling
Stage-4 complete by the same bar as Stage-3.

Stage-5 (self-written GPT-2 byte-level BPE tokenizer, ASCII scope) also
passes locally now — see "Self-written tokenizer (roadmap item 2): COMPLETE
(ASCII scope)" above. Same portability caveat as Stage-4: proven on this
machine with locally-fetched `vocab.json`/`merges.txt`, not yet from a clean
detached worktree the way Stage-3 was.

Stage-6 (generation correctness) also passes locally now — see "Generation
correctness (roadmap item 3): COMPLETE" above — with real text out of real
GPT-2 124M weights running through this project's own runtime, and 48/48
tokens matching real PyTorch greedy generation across 4 prompts token-for-
token, not just single-step logits. Same portability caveat as Stage-4/5;
same "bypasses the canonical planner/executor" boundary noted in roadmap
item 3 above — this generation path is a hand-written runtime-length C
forward pass anchored to, but not literally running through, the canonical
compile()/executor.

Stage-7 (canonical KV-cache integration) also passes its core claim locally
now — see "Canonical KV-cache integration (roadmap item 4): CORE CLAIM
COMPLETE" above — a new canonical op makes the cache real Tensor state, and
real GPT-2 124M decode through real compile()+executor calls against it
matches full recomputation at every generated step and matches real
PyTorch greedy generation. Speed not yet measured (item 5); the surrounding
per-position ops (MATMUL/BIAS_ADD/RESIDUAL/GELU) and prefill still use the
hand-written runtime-length kernels, deliberately, not yet brought under
canonical compile() calls of their own.

Current project status: **decoder-runtime op-matrix complete; real GPT-2 124M
block 0 complete and clean-worktree-portable; all 12 blocks + ln_f + tied LM
head logits match the oracle locally; a self-written ASCII-scope BPE
tokenizer matches an independent oracle locally (full Unicode
pre-tokenization classification explicitly out of scope); real end-to-end
generation works locally with 48/48 tokens matching real PyTorch greedy
generation across 4 prompts; a new canonical `CAUSAL_ATTENTION_CACHED` op
makes the KV cache real `Node`/`Tensor` state (available to, not yet
analyzed by, a future planner), and real GPT-2 124M decode through it
agrees with full recomputation within 1e-3 at every step (21/21) with
identical greedy tokens, and matches real PyTorch greedy generation;
Stage-4/5/6/7 portability to a clean worktree not yet proven; known open
gaps in the item-4 gates themselves (missing-model/fixture treated as a
silent PASS-via-skip rather than a required failure, no `--required` mode
like the block-0/full-model gates have, `greedy_reference.txt` parsed
without validating each read, no printed actual max-delta) are the
honest next fix, before either performance measurement (item 5) or the
llm.c cross-check noted under roadmap item 3's "сейчас" framing**.
