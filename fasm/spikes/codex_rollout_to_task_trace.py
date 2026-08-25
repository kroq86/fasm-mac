#!/usr/bin/env python3
"""Normalize a Codex rollout JSONL into the task-capsule spike event stream."""

import argparse
import json
import subprocess
import sys


def git(repo, *args, allow_failure=False):
    result = subprocess.run(("git", "-C", repo, *args), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode and not allow_failure:
        raise ValueError(result.stderr.strip() or "git command failed")
    return result.stdout.strip() if result.returncode == 0 else ""


def emit(events, kind, **fields):
    event = {"id": f"codex-{len(events) + 1:06d}", "kind": kind}
    event.update(fields)
    events.append(event)


def status_entries(repo):
    result = subprocess.run(("git", "-C", repo, "status", "--porcelain=v1", "-z"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip() or
                         "git status failed")
    raw = result.stdout.decode()
    parts = raw.split("\0") if raw else []
    for part in parts:
        if not part:
            continue
        code, path = part[:2], part[3:]
        if code == "??":
            state = "untracked"
        elif "D" in code:
            state = "deleted"
        else:
            state = "modified"
        yield path, state


def normalize(rollout, repo=None):
    events = []
    compacted = []
    with open(rollout, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_no}: invalid JSON") from error
            payload = record.get("payload", {})
            if not isinstance(payload, dict):
                continue
            record_type = record.get("type")
            subtype = payload.get("type")
            if record_type == "event_msg" and subtype == "user_message":
                text = payload.get("message")
                if isinstance(text, str) and text.strip():
                    emit(events, "goal", text=text.strip())
            elif record_type == "response_item" and subtype == "custom_tool_call":
                name = payload.get("name", "tool")
                emit(events, "command", text=f"tool:{name}",
                     status=payload.get("status", "requested"))
            elif record_type == "event_msg" and subtype == "patch_apply_end":
                status = "passed" if payload.get("success") else "failed"
                emit(events, "command", text="tool:apply_patch", status=status)
            elif record_type == "compacted":
                message = payload.get("message")
                if isinstance(message, str) and message:
                    compacted.append(message)
                elif any(isinstance(item, dict) and item.get("type") == "compaction" and
                         item.get("encrypted_content")
                         for item in payload.get("replacement_history", [])):
                    emit(events, "note", text="compaction:encrypted", status="observed")
    if repo:
        head = git(repo, "rev-parse", "--short", "HEAD")
        branch = git(repo, "branch", "--show-current")
        upstream = git(repo, "rev-parse", "--abbrev-ref", "@{upstream}",
                       allow_failure=True)
        fields = {"branch": branch, "head": head}
        if upstream:
            fields["upstream"] = upstream
        emit(events, "vcs", **fields)
        for path, state in status_entries(repo):
            emit(events, "file", path=path, state=state)
    return events, compacted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout")
    parser.add_argument("--repo")
    parser.add_argument("--summary-out")
    args = parser.parse_args()
    try:
        events, compacted = normalize(args.rollout, args.repo)
        for event in events:
            print(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        if args.summary_out:
            with open(args.summary_out, "w", encoding="utf-8") as stream:
                stream.write(compacted[-1] if compacted else "")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"codex-rollout-adapter: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
