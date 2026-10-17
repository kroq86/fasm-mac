# String-map replacement: localized regression fix

Date: 2026-10-18. User explicitly authorized fixing the reproduced defect and
adding regressions. Scope: hashmap_str_put_entry pointer preservation, small-map
mutation regressions and the existing MiniRedis consumers. Compiler contract,
resize, allocator, GC and error strategies were not changed.

## Problem, change and invariant

The original replacement/read probe reached UPDATE_BEGIN and terminated with
SIGSEGV. The new permanent mutation fixture likewise exited with subprocess
return code -11 against the unmodified core. Its source hash is identical in
the before/after records.

put_entry held its current entry address in rax while calling str_eq. That
function explicitly returns Boolean1/0 in rax. Both equal-key replacement and
the next-chain branch then used the Boolean as an address.

The fix saves rax on the stack before str_eq, tests the Boolean, and pops the
entry pointer back before branching. POP preserves the comparison flags. The
extra stack slot is balanced before either branch, leaving the existing saved
bucket-slot address at its previous stack position. Argument conventions,
callee-saved registers, map/entry layouts and allocation/ownership semantics
remain as before. No return-value guarantee is introduced; current put callers
ignore the returned rax.

This preserves three local obligations: comparison uses the Boolean result;
entry access uses the original pointer; the existing stack frame is restored
before insertion/update/cleanup. The narrow patch avoids replacing the search
with another full lookup or introducing an unrelated register/ABI convention.
It does not establish global memory safety.

## Evidence and regression coverage

Classification: implementation repair. Post-fix replacement behavior is
SUPPORTED in the executed domains; the original failure remains preserved as
pre-fix evidence, not rewritten as a passing experiment.

| Scenario | Before | After |
| --- | --- | --- |
| Permanent mutation fixture, same exact source | SIGSEGV (-11) | exit0, update checks passed |
| Original audit control | exit0 | exit0, CONTROL_OK |
| Original audit replacement/read | SIGSEGV (-11) | exit0, UPDATE_OK |

Permanent fixture: `fasm/tests/macos-smoke/hashmap_str_update.asm`.

- First insertion and same-key int replacement return the expected value.
- `foo` and `bff` must actually hash to the same bucket; the fixture checks it.
- A second colliding entry tests traversal; subsequent writes test both head
  and tail and preserve the neighboring value.
- Int→string, string→string, string→int replacements preserve type/value.
- Map size stays1 on initial replacement and2 through collision updates.
- Only two entries are used, below the default resize threshold.

`scripts/check_core_smoke.sh` now runs the existing map smoke, the new mutation
fixture, the existing MiniRedis script fixture and the full actual MiniRedis
REPL. The REPL sequence includes repeated SET, a colliding key, type changes,
INCR/DECR and DBSIZE, checked against its complete prompt/output transcript.

`scripts/check_miniredis_server.sh` checks repeated SET and type changes, retains
the colliding key, then restarts the server and requires latest values
`foo=baz`, `bff=chain`, `count=84`, `neg=-7`. Replay encounters repeated keys;
this is a tested restart scenario, not a universal crash-durability claim.

Required checks completed successfully:

```sh
bash scripts/check_core_smoke.sh
bash scripts/check_core_contracts.sh
bash scripts/check_machodoctor.sh
bash scripts/check_logknife.sh
bash scripts/check_fscan.sh
bash scripts/check_leetcode_examples.sh
bash scripts/check_miniredis_server.sh
bash -n scripts/check_core_smoke.sh scripts/check_miniredis_server.sh
git diff --check
```

The server check initially encountered sandbox denial at localhost bind. It
passed when run with approved local-network execution. No external network
service was used. A new wrapper initially expected nonexistent OK replies from
the unchanged old scripted fixture; source inspection showed its output is
only42/hello/2. That wrapper expectation was corrected; the dedicated mutation
fixture and product SET expectations were not weakened. The core fix required
one patch attempt.

## Caller audit and independent review

Direct pre-existing include consumers: MiniRedis, MiniRedis server, map smoke
and MiniRedis script smoke. All built and ran with the patch. Production call
sites include SET, new INCR/DECR paths, LOAD and server replay. The dedicated
new fixture is an additional consumer, not another product.

An independent Codex reviewer inspected the patch, every branch/stack path,
caller register/return assumptions, collision validity and CLI/server/replay
expectations; no blockers found. It performed no executions or edits. The
primary agent ran the checks and integrated the result. External Cursor was
unavailable; the previously blocked external Claude call was not retried.

## Trail, reproduction and limits

- [before.json](before.json): pre-fix core/fixture hashes, exact build/run
  commands and the new regression's SIGSEGV.
- [after.json](after.json): post-fix core hash, the identical fixture hash,
  rebuild/run records for the fixture and original audit probe.
- Supplemental original-audit sources and pre-fix evidence remain locally in
  `scratchpad/component_audit_20261018/`, outside this fix's commit. Their old
  source hashes describe the old version, not current core bytes. The committed
  permanent fixture and before/after records reproduce the defect independently.
- Executables are temporary x86_64 Mach-O builds run through Rosetta on this
  arm64 macOS host; the assembler reports classic1.73.35.

Reproduce the permanent checks with the commands above. To replay the pre-fix
failure, use a separate checkout of the recorded `git_head` in before.json,
copy in the permanent fixture from this commit, then build and run that fixture.
The supplemental old audit probe is not needed for this reproduction.

Known source-only concerns in resize remain OPEN. No claim is made about OOM,
GC/collection timing, arbitrary capacity, all map operations or all service
failure windows. Compiler acceptance findings and the negative agent gate are
unchanged. No architectural/API change. Commit and push were separately
authorized by the user after the checks completed.

Rollback: revert only this core patch, the two check-script changes, the new
mutation fixture and their status note. Preserve unrelated working-tree
changes and the original negative audit evidence.
