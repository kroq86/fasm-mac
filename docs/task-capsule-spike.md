# Task capsule spike

This spike tests whether an append-only agent trace can be compiled into a
small, verifiable continuation artifact. It is not an observability service or
a stable file format.

The input is JSON Lines with unique event IDs. State-like events are folded by
identity: the newest VCS state wins, and the newest status for each check and
path replaces its predecessor. Decisions, the goal, pending next actions, and
user-owned dirty files remain explicit. Every emitted claim carries its source
event ID. The capsule also records the SHA-256 digest and event count of the
source trace.

The byte budget is strict. Optional command/note context is admitted newest
first only when it fits. Required state is never silently truncated: the
compiler rejects an impossible budget instead. This is deliberately simpler
than semantic or learned ranking; those are unjustified until the structural
capsule proves useful.

Run the regression:

```sh
scripts/check_task_capsule_spike.sh
```

`task_capsule_verify.py` binds the emitted claims back to their source events
and independently checks the declared Git branch, HEAD, upstream, and dirty-file
states against a live workspace. The regression proves that a modified trace,
a changed claim, and stale live state cannot pass verification.

The fixture represents a real handoff shape from this repository. Its bounded
capsule retains the goal, Git state, user-owned dirty paths, latest regression
result, decisions, next action, and provenance. A raw tail with the same byte
budget loses the early goal and dirty-file ownership facts.

The spike only establishes lossless state folding against a weak raw-tail
baseline. It does **not** yet establish that another agent performs better with
the capsule than with a competent human/LLM summary. That requires a blinded
continuation evaluation before this can become a product direction.

## Codex rollout adapter

`codex_rollout_to_task_trace.py` reads the structured JSONL session format used
by Codex. The initial privacy boundary is intentionally narrow: it retains user
goals and tool names/statuses, but drops tool arguments, outputs, reasoning, and
patch contents. With `--repo`, it appends a fresh Git and dirty-worktree
observation instead of trusting session-start metadata that may already be
stale. The last built-in compaction message can be exported separately for an
eventual blinded comparison.

The checked-in fixture only locks the observed record shapes. Real rollouts are
processed locally into temporary files and must not be added to the repository.

On the first real Codex rollout (2,158 raw records), the adapter emitted 439
privacy-reduced events and compiled an 8,191-byte capsule that passed trace and
live-workspace verification. The rollout contained two compactions, but their
replacement summaries were stored as encrypted content and the public `message`
fields were empty. The adapter records only `compaction:encrypted`; it cannot
claim a textual quality comparison against an inaccessible built-in summary.
A fair A/B test therefore needs an explicitly supplied handoff baseline or two
fresh continuation runs, not rollout-file inspection.

## Continuation A/B harness

`task_capsule_render.py` renders the model-neutral JSON capsule as Markdown
without inventing new claims. Every state line retains its source event ID and
the integrity section exposes the bound trace digest.

`task_handoff_score.py` evaluates a continuation against a checked-in explicit
rubric. It counts required fact recovery and separately penalizes unsafe actions
and already-completed work. The regression locks the scorer with one complete
answer and one deliberately unsafe answer; it does not use those fixtures as
evidence that the capsule improves a real model. A genuine A/B result still
requires fresh, isolated continuations receiving identical task prompts.

### A/B run 1: retry policy

The first real continuation comparison used two isolated copies of a new C
project. Both participants received the same request and source tree. A received
a concise conventional `HANDOFF.md`; B received the rendered verified capsule.
Neither participant inherited the parent conversation or could inspect the
other workspace.

Both implementations independently chose overflow-safe saturating shifts,
passed the visible test, passed a withheld large-attempt test, changed only
`retry_policy.c`, and preserved the user-owned untracked `local.env`. The result
is a tie: `visible=2/2`, `hidden=2/2`, `preserved=2/2`. This run demonstrates
that the capsule is sufficient, but provides no evidence that it is better than
a competent concise handoff on a small task. A later run must introduce enough
state transitions and superseded evidence to test the claimed advantage.

### A/B run 2: superseded routing state

The second comparison added an obsolete filename, a reverted implementation, a
superseded retry limit, failed-to-passed check history, a new final goal, and a
user-owned header overlapping the implementation contract. The conventional
handoff described that chronology explicitly; the capsule lowered it to current
state and omitted the superseded attempt-5 rule and failed check.

The result was again a tie. Both continuations changed only `route_result.c`,
ignored the obsolete `retry_route.c`, preserved the untracked user header, and
passed both visible and withheld boundary tests: `visible=2/2`, `hidden=2/2`,
`preserved=2/2`, `stale-file-untouched=2/2`.

Two runs therefore show sufficiency but no model-quality advantage over a
competently written handoff. The remaining plausible benefit is automation at
scale: generating reliable state without requiring a human to write that good
handoff. Further A/B work should compare automatic summaries over long traces,
not keep inventing increasingly elaborate small coding puzzles.
