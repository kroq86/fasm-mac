# KV-handoff manuscript

Read [paper.md](paper.md). English research draft, not submitted to arXiv.
No new training or held-out experiments were run for this manuscript.

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
are included in the source ledger; no new experiments were run for this revision.

## Reproduce the integrated checkpoint

```sh
bash scripts/build-kv-handoff.sh
python3 scripts/kv_handoff.py --assets /absolute/path/to/bundle \
  --sequence AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB
bash scripts/check_kv_handoff.sh /absolute/path/to/bundle
```

See [native preparation and limitations](../../scratchpad/kv_transfer_audit_20260913/NATIVE_HANDOFF.md).
The large base-model/reference bundle is external, not committed. The ordering
adapter itself is already tracked. The separate lookup adapter is not saved;
the article's lookup table comes from the tracked script/result, not a new replay.

## Remaining submission decisions

- Author names/affiliations and author approval: not invented by the assistant.
- Format export: this commit supplies the complete Markdown manuscript, not a
  compiled LaTeX/PDF submission package.
- Scope: small-model retrospective case study plus native reproduction, not a
  new-method claim. The related-work section explicitly cites the closest causal
  audit, not merely architecture papers.
- Missing validation is stated in the paper: lookup checkpoint replay, seed
  variance, independent ordering test, and matched efficiency baselines.
  These are not results implicitly promised by the existing evidence.
- Citation pages were checked on 2026-09-13; competing methods were not rerun.

Do not strengthen the abstract by hiding these boundaries. Further experiments
require a separate decision; writing this draft does not authorize them.
