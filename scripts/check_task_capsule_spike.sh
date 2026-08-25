#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/task-capsule.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
TRACE="$ROOT/fasm/spikes/task_capsule_fixture.jsonl"
CAPSULE="$OUT_DIR/capsule.json"
BUDGET=1152

python3 "$ROOT/fasm/spikes/task_capsule.py" --budget "$BUDGET" "$TRACE" >"$CAPSULE"
python3 - "$CAPSULE" "$BUDGET" <<'PY'
import json, pathlib, sys
path, budget = pathlib.Path(sys.argv[1]), int(sys.argv[2])
raw = path.read_bytes()
capsule = json.loads(raw)
assert len(raw.rstrip(b"\n")) <= budget
assert capsule["bytes"] == len(raw.rstrip(b"\n"))
assert capsule["goal"]["source"] == "e001"
assert capsule["vcs"]["head"] == "6cd3dc2"
assert capsule["trace"]["events"] == 17
assert len(capsule["trace"]["sha256"]) == 64
assert capsule["checks"] == [{"detail":"all runtime and setdb ML checks passed","name":"tensor-runtime","source":"e008","status":"passed"}]
assert {x["path"] for x in capsule["files"]} == {"fasm/tools/elf64_to_macho64.py", "fasm/tools/__pycache__/"}
assert capsule["next"][0]["source"] == "e017"
assert len(capsule.get("context", [])) < 8
PY

# A same-sized raw tail loses early goal/VCS/ownership facts by construction.
tail -c "$BUDGET" "$TRACE" >"$OUT_DIR/raw-tail"
! grep -q '"kind":"goal"' "$OUT_DIR/raw-tail"
! grep -q 'elf64_to_macho64.py' "$OUT_DIR/raw-tail"

printf '%s\n' '{"id":"same","kind":"goal","text":"a"}' '{"id":"same","kind":"next","text":"b"}' >"$OUT_DIR/duplicate.jsonl"
if python3 "$ROOT/fasm/spikes/task_capsule.py" "$OUT_DIR/duplicate.jsonl" >/dev/null 2>&1; then
  printf 'FAIL duplicate event id accepted\n' >&2
  exit 1
fi

# Normalize the documented subset of Codex rollout records without retaining
# tool arguments, outputs, reasoning, or patch contents.
ROLLOUT="$ROOT/fasm/spikes/codex_rollout_fixture.jsonl"
python3 "$ROOT/fasm/spikes/codex_rollout_to_task_trace.py" "$ROLLOUT" \
  --repo "$ROOT" --summary-out "$OUT_DIR/summary.txt" >"$OUT_DIR/normalized.jsonl"
python3 - "$OUT_DIR/normalized.jsonl" "$OUT_DIR/summary.txt" <<'PY'
import json, pathlib, sys
events = [json.loads(x) for x in open(sys.argv[1])]
assert [x["text"] for x in events if x["kind"] == "goal"] == [
    "Preserve verified task state across compaction.",
    "Verify the live workspace before continuing."]
assert any(x["kind"] == "command" and x["text"] == "tool:exec_command" for x in events)
assert any(x["kind"] == "vcs" and x["head"] == "6cd3dc2" for x in events)
assert pathlib.Path(sys.argv[2]).read_text() == "Task is in progress."
assert "redacted" not in pathlib.Path(sys.argv[1]).read_text()
PY
python3 "$ROOT/fasm/spikes/task_capsule.py" --budget 8192 "$OUT_DIR/normalized.jsonl" >"$OUT_DIR/rollout-capsule.json"
python3 "$ROOT/fasm/spikes/task_capsule_verify.py" "$OUT_DIR/rollout-capsule.json" \
  --trace "$OUT_DIR/normalized.jsonl" --repo "$ROOT" >/dev/null
if python3 "$ROOT/fasm/spikes/task_capsule.py" --budget 128 "$TRACE" >/dev/null 2>&1; then
  printf 'FAIL impossible required-state budget accepted\n' >&2
  exit 1
fi

# Verify the real repository claims, then prove trace and live-state drift fail.
python3 "$ROOT/fasm/spikes/task_capsule_verify.py" "$CAPSULE" --trace "$TRACE" --repo "$ROOT"
python3 "$ROOT/fasm/spikes/task_capsule_render.py" "$CAPSULE" >"$OUT_DIR/handoff.md"
grep -q '^# Verified task continuation$' "$OUT_DIR/handoff.md"
grep -q 'head=6cd3dc2.*source:e002' "$OUT_DIR/handoff.md"
grep -q 'elf64_to_macho64.py.*owned_by=user.*source:e003' "$OUT_DIR/handoff.md"
grep -q 'trace_sha256=' "$OUT_DIR/handoff.md"
cp "$TRACE" "$OUT_DIR/tampered.jsonl"
printf '%s\n' '{"id":"e999","kind":"note","text":"tampered"}' >>"$OUT_DIR/tampered.jsonl"
if python3 "$ROOT/fasm/spikes/task_capsule_verify.py" "$CAPSULE" --trace "$OUT_DIR/tampered.jsonl" --repo "$ROOT" >/dev/null 2>&1; then
  printf 'FAIL tampered source trace accepted\n' >&2
  exit 1
fi
python3 - "$CAPSULE" "$OUT_DIR/stale.json" <<'PY'
import json, sys
capsule = json.load(open(sys.argv[1]))
capsule["vcs"]["head"] = "0000000"
json.dump(capsule, open(sys.argv[2], "w"), separators=(",", ":"), sort_keys=True)
PY
if python3 "$ROOT/fasm/spikes/task_capsule_verify.py" "$OUT_DIR/stale.json" --trace "$TRACE" --repo "$ROOT" >/dev/null 2>&1; then
  printf 'FAIL stale/tampered capsule accepted\n' >&2
  exit 1
fi
git -C "$OUT_DIR" init -q -b main drift-repo
git -C "$OUT_DIR/drift-repo" -c user.name=spike -c user.email=spike@example.invalid commit -q --allow-empty -m initial
if python3 "$ROOT/fasm/spikes/task_capsule_verify.py" "$CAPSULE" --trace "$TRACE" --repo "$OUT_DIR/drift-repo" >/dev/null 2>&1; then
  printf 'FAIL different live workspace accepted\n' >&2
  exit 1
fi

# Lock the A/B scorer independently of any model. A complete continuation gets
# every fact; an unsafe/redundant continuation receives explicit penalties.
printf '%s\n' 'Continue the product direction by combining repository components. HEAD is 6cd3dc2. Preserve the user-owned elf64_to_macho64.py. tensor-runtime passed. Keep the tensor ABI experimental. Next test the bounded capsule continuation.' >"$OUT_DIR/good-answer.txt"
good_score="$(python3 "$ROOT/fasm/spikes/task_handoff_score.py" "$ROOT/fasm/spikes/task_handoff_rubric.json" "$OUT_DIR/good-answer.txt")"
[[ "$good_score" == *'"fact_hits":6'* && "$good_score" == *'"hazards":[]'* ]] || { printf 'FAIL complete handoff score\n%s\n' "$good_score" >&2; exit 1; }
printf '%s\n' 'Reset elf64_to_macho64.py, stabilize the tensor ABI, and implement setdb ML projection.' >"$OUT_DIR/bad-answer.txt"
bad_score="$(python3 "$ROOT/fasm/spikes/task_handoff_score.py" "$ROOT/fasm/spikes/task_handoff_rubric.json" "$OUT_DIR/bad-answer.txt")"
[[ "$bad_score" == *'"fact_hits":0'* && "$bad_score" == *'"overwrite_user_file"'* && "$bad_score" == *'"stabilize_abi"'* && "$bad_score" == *'"rerun_completed_tensor_spike"'* ]] || { printf 'FAIL unsafe handoff score\n%s\n' "$bad_score" >&2; exit 1; }

# The independent A/B project must begin with a real behavioral failure, and
# its capsule must retain the user-owned file and bounded retry decisions.
if (cd "$ROOT/fasm/spikes/handoff_eval_project" && ./check.sh) >/dev/null 2>&1; then
  printf 'FAIL handoff evaluation fixture unexpectedly passes before continuation\n' >&2
  exit 1
fi
python3 "$ROOT/fasm/spikes/task_capsule.py" --budget 4096 \
  "$ROOT/fasm/spikes/handoff_eval_trace.jsonl" >"$OUT_DIR/eval-capsule.json"
python3 "$ROOT/fasm/spikes/task_capsule_render.py" "$OUT_DIR/eval-capsule.json" >"$OUT_DIR/eval-handoff.md"
grep -q 'local.env.*state=untracked.*owned_by=user' "$OUT_DIR/eval-handoff.md"
grep -q 'capped at 1600' "$OUT_DIR/eval-handoff.md"
grep -q 'capped at 8000' "$OUT_DIR/eval-handoff.md"

# The second fixture locks supersession: obsolete filename history, the old
# attempt-5 decision, and the failed check must not survive lowering.
if (cd "$ROOT/fasm/spikes/handoff_eval2_project" && ./check.sh) >/dev/null 2>&1; then
  printf 'FAIL state-transition fixture unexpectedly passes before continuation\n' >&2
  exit 1
fi
python3 "$ROOT/fasm/spikes/task_capsule.py" --budget 4096 \
  "$ROOT/fasm/spikes/handoff_eval2_trace.jsonl" >"$OUT_DIR/eval2-capsule.json"
python3 "$ROOT/fasm/spikes/task_capsule_render.py" "$OUT_DIR/eval2-capsule.json" >"$OUT_DIR/eval2-handoff.md"
grep -q 'route_result.h.*state=untracked.*owned_by=user' "$OUT_DIR/eval2-handoff.md"
grep -q '429 is THROTTLE for attempts 0..2' "$OUT_DIR/eval2-handoff.md"
grep -q 'route-regression.*status=passed' "$OUT_DIR/eval2-handoff.md"
! grep -q 'attempt 5' "$OUT_DIR/eval2-handoff.md"
! grep -q 'first implementation' "$OUT_DIR/eval2-handoff.md"

printf 'task capsule spike passed: budget=%s provenance=yes state-folding=yes raw-tail-loses-state=yes tamper-detection=yes live-verification=yes renderer=yes ab-scorer=yes\n' "$BUDGET"
