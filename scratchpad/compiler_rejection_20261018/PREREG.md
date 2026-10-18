# Local compiler rejection repair

Scope: `fasm/spikes/tensor_semantic_compiler.h`; retain its existing node,
dispatch, schedule and return-code contracts. This is an engineering repair,
not a new architecture, universal graph verifier or model-capability experiment.

## Claims and decision

- C1: unsupported opcodes are rejected before dispatch-table indexing/emission.
  Prior evidence: audit accepted `OP_COUNT`; the same graph with MATMUL executes
  2*3=6 and a bad shape is rejected. Alternative: the rejection only follows
  from malformed shapes, indices or buffers. Matched cases change only opcode
  to `OP_COUNT` and `UINT32_MAX`; valid inputs/shapes remain fixed.
- C2: compile rejects a graph whose derived gradient needs would schedule a
  missing backward handler. The existing cached-attention contract explicitly
  excludes backward. Alternative: rejection is caused by invalid cache metadata
  or a blanket ban on graphs containing trainable/retained nodes. Controls:
  validate must first accept each target; inference executes; an unrelated
  PARAM leaf remains accepted. Interventions: PARAM/RETAIN_GRAD on QKV, cache,
  cached output, and an upstream node independently. Preserve graph/buffers.
- C3: the local guards preserve supported inference/training. Its evidence is
  the executed positive checks and existing caller regressions, not C1/C2 alone.

C1 and C2 are independent rejection obligations. C1/C2 plus C3 are necessary
for accepting this local patch; none proves general memory or graph safety.

## Procedure, metrics, criteria, budget

First add an executable regression and run it on the unchanged header. Save
stdout/stderr/exit code. Then add guards using existing `-1` compile rejection,
rerun the identical regression, and inspect the actual diff independently.
Never execute an accepted unsupported schedule in the pre-patch run.

Support C1/C2 iff all designated targets reject, their valid controls pass,
and rejection occurs before writing steps/contexts/output count. Refute if a
target is accepted; invalid if compilation/instrument/control fails. Support C3
only for executed gates; missing optional assets are skips, not passes.

Regression scope: valid MATMUL and malformed-shape control; unknown opcode
sentinels; cached inference (including cache append/output); each gradient
source above; unrelated PARAM control; existing MLP/Transformer/CNN/POOL checks,
cached-attention differential, and real GPT-2 CLI/cached-generation gates with
the existing pinned model. Sanitizers on native-C regression supplement the
x86_64 assembly-executor run.

Budget: one local repair plus at most one correction after concrete feedback;
stop after two failed fixes, a contract/architecture decision, or 30 minutes.
Failure retains the old evidence and leaves repair incomplete; no widening to
new operators, caching protocol, executor semantics or proposer experiments.

Baseline: unchanged repository header and existing enum-indexed FWD/BWD dispatch.
No novel mechanism claimed. Record compiler/host, commands, raw output and diff;
original audit evidence stays immutable. Rollback removes only this patch and
its new regression files; preserve pre-existing user changes.
