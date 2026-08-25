#!/usr/bin/env python3
"""Verify a task capsule against its source trace and live Git workspace."""

import argparse
import hashlib
import json
import subprocess
import sys


def git(repo, *args):
    result = subprocess.run(("git", "-C", repo, *args), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def claims(value):
    if isinstance(value, dict):
        if "source" in value:
            yield value
        for child in value.values():
            yield from claims(child)
    elif isinstance(value, list):
        for child in value:
            yield from claims(child)


def dirty_state(repo, path):
    lines = git(repo, "status", "--porcelain=v1", "--", path).splitlines()
    if not lines:
        return "clean"
    code = lines[0][:2]
    if code == "??":
        return "untracked"
    if "D" in code:
        return "deleted"
    return "modified"


def verify(capsule, trace_path, repo):
    errors = []
    raw = open(trace_path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    trace_meta = capsule.get("trace", {})
    if trace_meta.get("sha256") != digest:
        errors.append("trace digest mismatch")
    events = {}
    for line_no, line in enumerate(raw.decode().splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("id") in events:
            errors.append(f"duplicate trace event id {event.get('id')}")
        events[event.get("id")] = event
    if trace_meta.get("events") != len(events):
        errors.append("trace event count mismatch")
    ignored = {"source"}
    for claim in claims(capsule):
        source = events.get(claim["source"])
        if source is None:
            errors.append(f"missing source event {claim['source']}")
            continue
        for key, value in claim.items():
            if key not in ignored and source.get(key) != value:
                errors.append(f"claim {claim['source']} field {key} differs from trace")
    vcs = capsule.get("vcs")
    if vcs:
        live = {"branch": git(repo, "branch", "--show-current"),
                "head": git(repo, "rev-parse", "--short", "HEAD")}
        try:
            live["upstream"] = git(repo, "rev-parse", "--abbrev-ref", "@{upstream}")
        except ValueError:
            live["upstream"] = ""
        for key in ("branch", "head", "upstream"):
            if key in vcs and vcs[key] != live[key]:
                errors.append(f"live vcs {key}: capsule={vcs[key]} actual={live[key]}")
    for item in capsule.get("files", []):
        actual = dirty_state(repo, item["path"])
        if actual != item["state"]:
            errors.append(f"live file {item['path']}: capsule={item['state']} actual={actual}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capsule")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    try:
        with open(args.capsule, encoding="utf-8") as stream:
            capsule = json.load(stream)
        errors = verify(capsule, args.trace, args.repo)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        print(f"task-capsule-verify: {error}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"FAIL {error}", file=sys.stderr)
        return 1
    print("task capsule verified: trace=yes provenance=yes workspace=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
