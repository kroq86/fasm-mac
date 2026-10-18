# Independent critic record

Identity: internal Codex subagent `/root/compiler_rejection_critic`, fresh context.
Not Claude or Cursor. Read-only source/diff inspection; no external export, edit,
build or test. Provider token counters unavailable. Primary executed the gates.

Initial review identified the necessary exclusions and control:

- Bound opcode before indexing FWD; LEAF intentionally lacks FWD/BWD.
- Derive needs first, then reject nonleaf missing BWD before schedule writes.
- Include cached output's own RETAIN_GRAD and transitive needs.
- Cache PARAM may accidentally fail earlier optimizer-metadata validation;
  require validate==0 to isolate missing backward.
- Product cached callers use INPUT/INPUT/TEMP. The old semantic-training gate
  tests a different compiler; use merged training and actual cached callers.

Actual-diff review found no blocking implementation defect. The regression's
cache-PARAM setup satisfies metadata validation and the saved pre-patch run
accepts it, so that rejection is not a false positive. Strongest remaining
coverage gap: successful nonleaf backward is absent from the new local check.
Primary subsequently ran the existing merged MLP/Transformer/CNN/POOL training
gates, including finite-difference gradients and optimization.

The critic flagged an overbroad comment, "Invalid or unsupported graphs return
-1". Primary changed it to "Validation or unsupported-dispatch failures return
-1". This is a documentation correction; the three added executable guard lines
were unchanged. Capacity failure (-2) can still partially write schedules.

Open scope: general operand validation, malformed storage, postcompile metadata
mutation and universal executor safety. No independent empirical rerun claimed.
