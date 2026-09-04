# Verified state-transition agent: preregistered spike ladder

## Claim boundary

The candidate claim is not that an LLM becomes deterministic or that typed
execution is novel. The testable claim is narrower: separating a stochastic
proposal from a typed, transactional, independently verified state transition
can reduce invalid actions without winning merely by abstaining.

## Frozen order

1. **Instrument sensitivity.** A tiny world-state executor must commit valid
   transitions atomically, roll back late failures, replay byte-identically,
   and fail a mutation where invariant checking is disabled.
2. **Proposer feasibility.** On a frozen request set, measure whether the
   language frontend can emit the closed typed schema. Report syntax/type/
   semantic validity separately. If valid coverage is below 70%, do not claim
   an end-to-end agent result and do not fine-tune post hoc.
3. **Matched three-arm comparison.** Give direct-action, tool-schema and
   verified-transaction arms the identical frozen proposals. Primary endpoint:
   task success with zero invariant violations. Also report coverage,
   accepted-only success and refusal rate; a verifier may not win solely by
   refusing more work.
4. **Held-composition test.** Freeze atomic actions during development and hold
   out combinations of battery, temperature, critical load and generator
   failure. Compare against an exact classical policy baseline. If the exact
   policy wins, record `NO_ADVANTAGE_OVER_CLASSICAL`.

## Stop conditions

- No new world complexity until the preceding gate passes.
- No threshold/task retuning after seeing held-composition results.
- GPT-2 quality and executor safety are separate endpoints.
- A deterministic executor beside an LLM is not described as deterministic
  reasoning inside the neural model.
- Two failed attempts at the same gate close the branch for this round.

