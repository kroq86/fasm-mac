# Compiler rejection repair — 2026-10-18

Verdict: **SUPPORTED engineering repair** for two dispatch acceptance holes in
the canonical compiler. This is not a general graph-safety proof or a model
quality/performance result. Preregistered boundary: [PREREG.md](PREREG.md).

## Actual evolution

| Version | Actual artifact and check | Outcome / next decision |
| --- | --- | --- |
| Before | Unchanged header, new identical regression | 21 checks, 10 failures; valid controls pass. Fix acceptance boundary. |
| Repair | Bounds/FWD guard in validate; needs-driven BWD guard before emission | 21 checks, zero failures. Validate integration rather than declaring general safety. |
| Critic feedback | Read-only independent review of header/diff and executable cases | No blocking code defect; narrow an overbroad comment and require successful supported training. |
| Integration | Existing training, op, cached-inference and actual GPT-2 gates | 16/16 pass, no skips; native-C ASan/UBSan pass. Stop at completed local repair. |

The patch adds three executable guard lines. It preserves the existing -1
compile rejection, LEAF dispatch exemptions and supported operator semantics.
It does not implement a cached-attention gradient. New rejection cases leave
steps, contexts and output count unchanged; capacity failure (-2) retains its
old partial-write behavior. The new regression is included in the existing
`check_tensor_runtime_spikes.sh` list; that entire aggregate was not rerun here.

## Claims, controls and evidence

- **C1 SUPPORTED:** change only a valid MATMUL opcode to OP_COUNT or UINT32_MAX;
  validate and compile now reject. The executable never dispatches unsupported
  schedules, including in the pre-patch run. Raw paired results: `before.*`,
  `after.*`. The bounds expression precedes table indexing by C short-circuit
  semantics; this is code-level reasoning, not exhaustive runtime exploration.
- **C2 SUPPORTED:** cached INPUT/INPUT inference executes and appends the expected
  K/V/output. PARAM/RETAIN_GRAD on QKV or cache, output RETAIN_GRAD, and a PARAM
  through RELU all pass shape/metadata validation but now reject at compile.
  An unrelated PARAM remains accepted and its value stays unchanged. The cache
  PARAM fixture explicitly satisfies optimizer metadata to exclude that confound.
- **C3 SUPPORTED only for executed gates:** MLP, Transformer, CNN and POOL retain
  finite-difference gradient checks, scheduled updates and their expected task
  outcomes. Fusion and eight decoder/cache op gates also pass. GPT-2 CLI matches
  its exact expected continuation and invalid-input checks. Cached/full generation
  agrees on every tested step (three prompts, seven decode steps each, max logit
  delta 0), and the reference greedy sequence matches. Full 12-block differential
  passes all four boundaries; logits max_abs=0.000289917, zero failed elements.
  The existing full gate also repeats with byte-identical reported output.

C1 and C2 are independent obligations, not evidence for C3. C3 required its own
actual executions. [REVIEW.md](REVIEW.md) records the critic's source-level
findings, not an independent empirical rerun.

`regressions.json` contains exact commands and results; per-gate stdout/stderr
retain numerical output. Rejection regression and existing gates execute x86_64
with the assembly executor on this arm64 macOS host. ASan/UBSan instead executes
native arm64 with the existing C dispatch-loop alternative; it does not sanitize
the x86 assembly executor. Compiler/platform/flags: `asan.json`.

All 32 direct C include consumers compile to objects: 30 with their source/default
macros, two GPT-2 callers with the real geometry from their checked-in build scripts.
The first generic sweep's two explicit #error failures are preserved in
`consumer-builds.json`; it had omitted required build macros and is not evidence
of a source regression. Corrected commands/results are in
`consumer-builds-real-shapes.json`. Compilation alone is not an execution result;
dataset-dependent workloads and the separate handoff executable were not rerun.

## Limits and stop

General operand/buffer validation, postcompile metadata changes, allocation
capacity and universal runtime refinement remain open. No core assembly,
ExecStep ABI, cache state protocol, model weights, training regime or proposer
gate changed. Confident wrongness/local success bias would be claiming that
21 checks or agent agreement establish full compiler safety; neither does.

No runtime failed-fix retry was needed. The remaining gap identified by the
critic (supported training) was resolved by existing executed gates. Stop because
the bounded repair meets its bar, not because every component is now mature.
Further component changes remain separately scoped.

## Reproduction and change trail

From repository root:

```sh
scripts/check_tensor_compiler_rejection.sh
scripts/check_tensor_merged_mlp_spike.sh
scripts/check_tensor_merged_transformer_spike.sh
scripts/check_tensor_merged_cnn_spike.sh
scripts/check_tensor_merged_pool_spike.sh
scripts/check_tensor_causal_attention_cached_spike.sh
GPT2_SAFETENSORS=/path/to/pinned/model.safetensors scripts/check_tensorctl_gpt2.sh
scripts/check_tensor_gpt2_cached_generation_spike.sh --required --model /path/to/pinned/model.safetensors
scripts/check_tensor_gpt2_full_spike.sh --required --model /path/to/pinned/model.safetensors
git diff --check
```

Model SHA256 is 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707;
full/cached gates verify existing reference fixture hashes. `provenance.json`
records exact paths, source/check hashes, compiler and parent revision; `patch.diff`
contains the code and gate diff with zero context (use `git apply --unidiff-zero`
when applying it). Its representation was regenerated during commit preparation
to avoid blank context lines being flagged as trailing whitespace in the stored
patch file. Source and raw measurement results are unchanged. No random
inputs/seeds or thresholds changed.

Rejected alternatives: leave the two known acceptance holes; ban any graph
containing PARAM plus cache (breaks the unrelated branch); add cached backward
or redesign op traits/metadata (outside this repair). Rollback removes only the
recorded local patch/new regression, preserving pre-existing STATUS/Python edits.
No commit or push had been executed when this repair result was recorded.
