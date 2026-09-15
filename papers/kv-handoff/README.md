# KV-handoff manuscript

Read [paper.md](paper.md). English research draft, not submitted to arXiv.
No new training was run for the originally reported experiments (ordering,
lookup, two-key, XOR). A later revision added, for the ordering task only, six
new inference-only checks against the same frozen checkpoint (moment-matched
random cache, a multi-configuration re-prefill/cache-reuse latency comparison,
attention-output cosine for the K-vs-V asymmetry, an alternate-symbol
template-family check, a sequence-length check, and a five-seed
adapter-initialization variance check; all Section 5.1) and three new training
runs, each explicitly labeled and evidenced separately from the historical
results: a converged rerun under the corrected XOR half-swap pairing (Section
5.3, verdict NOT_SUPPORTED for a diagnosed reason), a same-seed lookup retrain
(Section 5.2, verdict reproduces, exact accuracy does not), and the five-seed
ordering variance runs mentioned above. All are disclosed below and in the
paper, not folded silently into the original claims.

## Check the numbers and source identity

From the repository root:

```sh
python3 papers/kv-handoff/evidence.py --check
```

`evidence.json` records selected original metrics, their source paths and SHA-256
hashes, plus code/checkpoint identity. A passed check does not validate the
experimental design or recompute bootstrap intervals. The manuscript identifies
original-dev, stored-test, and native-reproduction evidence separately.

The manuscript's experiment summary table separates adapters and evaluation
populations. Appendix A maps the broader research questions to measured,
unsupported or untested claims without turning the exploratory inventory into
a sequence of established capabilities. Scale/noise/weak-conflict measurements
are included in the source ledger; a moment-matched random-cache control, a
multi-configuration re-prefill/cache-reuse latency comparison (four
threads/batch-size configurations, both fresh-cost and sunk-cost sender
accounting), an attention-output cosine diagnostic, an alternate-symbol
template-family check, a sequence-length check, and a five-seed variance check
were added in a later revision, all as dedicated
`phase1_kv_cache_minigunpoint_32token_*.py` scripts running inference-only
(the variance check trains fresh adapters but reuses the same fixed dev split)
against the same frozen ordering checkpoint's dataset. A second,
independent latency/cache-reuse comparison for the lookup task
(`phase1_kv_cache_sql_lookup_binding_multiquery_latency.py`, distinct queries
against one cached table, its own freshly trained adapter) and a direct
timing of the native `tensor-kv-handoff` executable itself
(`scripts/bench_kv_handoff_native_latency.py`) were added alongside it.

Correction and fix: the historical XOR `half_swap` design paired doc i's half1
with doc (i+1 mod n)'s half2, which produced 32 class-1 targets. The 14/32
score is preserved as a historical artifact, but was not a clean negative
result from a balanced composition test. The pairing bug has since been fixed
in the tracked source (offset changed to i+2, with a runtime balance
assertion). Run `python3 papers/kv-handoff/check_xor_design.py` to verify the
corrected pairing is now balanced (16/16), not merely to diagnose the old
defect; the ledger also runs this diagnostic and reports the live, current
state. Historical result verdict strings remain verbatim evidence, not the
corrected interpretation. A new training run under the corrected pairing
(different data scale, one seed, 800 epochs) converged and is reported in the
paper as `KV_COMPOSITIONALITY_XOR_NOT_SUPPORTED` for a diagnosed reason
(half-only recall nearly matches correct-condition accuracy), not as a
repaired replacement for the historical score, which remains separately
evidenced as invalid-as-computed.

## Reproduce the integrated checkpoint

```sh
bash scripts/build-kv-handoff.sh
python3 scripts/kv_handoff.py --assets /absolute/path/to/bundle \
  --sequence AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB
bash scripts/check_kv_handoff.sh /absolute/path/to/bundle
```

See [native preparation and limitations](../../scratchpad/kv_transfer_audit_20260913/NATIVE_HANDOFF.md).
The large base-model/reference bundle is external, not committed. A later
revision also timed the native executable directly
(`scripts/bench_kv_handoff_native_latency.py`, result stored alongside
`native_gate.log`): roughly 6.7-7.2 s per cold invocation on the arm64 host
used for this repository, about 10x the PyTorch-reference-environment cache-
transfer latency in Section 5.1, because the executable is x86-64-only
(FASM assembly, not portable across ISAs) and runs under Rosetta 2 here,
re-fingerprinting the ~900 MB external bundle from a cold process every
call. No arm64 build exists or was attempted. The ordering adapter itself
is already tracked. The separate lookup adapter is not saved;
the article's primary lookup table comes from the tracked script/result. A
later revision reran the unmodified lookup script from scratch (same seed) as
a checkpoint-replay attempt: the qualitative verdict reproduced, the exact
accuracy (16/32 vs. the historical 20/32) did not, reported in the paper
(Section 5.2) as a training-seed-sensitivity finding, not a corrected number.

## Remaining submission decisions

- Author names/affiliations and author approval: not invented by the assistant.
- Format export: this commit supplies the complete Markdown manuscript, not a
  compiled LaTeX/PDF submission package.
- Scope: small-model retrospective case study plus native reproduction, not a
  new-method claim. The related-work section explicitly cites the closest causal
  audit, not merely architecture papers.
- Missing validation is stated in the paper: a lookup checkpoint replay attempt
  now exists (verdict reproduces, exact accuracy does not), a five-seed
  adapter-initialization variance check for the ordering task exists (31-32/32
  across all five seeds), and a PyTorch-reference-environment latency
  comparison now exists under both fresh-cost and sunk-cost sender accounting
  (direction robust, magnitude configuration-dependent). Seed variance for
  lookup/two-key/XOR beyond the single retrains already reported, an
  independent ordering test, and a matched *native*-path efficiency comparison
  remain absent. These are not results implicitly promised by the existing
  evidence.
- Citation pages were checked on 2026-09-13; competing methods were not rerun.

Do not strengthen the abstract by hiding these boundaries. Further experiments
require a separate decision; writing this draft does not authorize them.
