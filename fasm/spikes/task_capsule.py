#!/usr/bin/env python3
"""Compile an append-only agent trace into a bounded resumable-state capsule."""

import argparse
import hashlib
import json
import sys


class TraceError(ValueError):
    pass


def load_events(path):
    events = []
    seen = set()
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceError(f"line {line_no}: invalid JSON: {error.msg}") from error
            if not isinstance(event, dict) or not isinstance(event.get("id"), str):
                raise TraceError(f"line {line_no}: event requires a string id")
            if event["id"] in seen:
                raise TraceError(f"line {line_no}: duplicate event id {event['id']}")
            if not isinstance(event.get("kind"), str):
                raise TraceError(f"line {line_no}: event requires a string kind")
            seen.add(event["id"])
            events.append(event)
    return events


def sourced(event, *fields):
    item = {field: event[field] for field in fields if field in event}
    item["source"] = event["id"]
    return item


def compile_state(events):
    state = {"schema": "task-capsule/v0", "checks": [], "files": [],
             "decisions": [], "next": [], "context": []}
    files = {}
    checks = {}
    decisions = {}
    context = []
    for event in events:
        kind = event["kind"]
        if kind == "goal" and "text" in event:
            state["goal"] = sourced(event, "text")
        elif kind == "vcs" and "head" in event:
            state["vcs"] = sourced(event, "branch", "head", "upstream")
        elif kind == "file" and "path" in event and "state" in event:
            files[event["path"]] = sourced(event, "path", "state", "owned_by")
        elif kind == "check" and "name" in event and "status" in event:
            checks[event["name"]] = sourced(event, "name", "status", "detail")
        elif kind == "decision" and "key" in event and "text" in event:
            decisions[event["key"]] = sourced(event, "key", "text")
        elif kind == "next" and "text" in event:
            state["next"].append(sourced(event, "text"))
        elif kind in ("command", "note"):
            context.append(sourced(event, "text", "status"))
    state["files"] = list(files.values())
    state["checks"] = list(checks.values())
    state["decisions"] = list(decisions.values())
    state["context"] = context
    return state


def encoded(state):
    return json.dumps(state, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def fit_budget(state, budget):
    # Context is optional and newest-first. Explicit state is never silently cut.
    context = state.pop("context")
    state["bytes"] = 0

    def settle_size():
        while True:
            size = len(encoded(state))
            if state["bytes"] == size:
                return size
            state["bytes"] = size

    if settle_size() > budget:
        raise TraceError(f"required state exceeds {budget}-byte budget")
    for item in reversed(context):
        state.setdefault("context", []).insert(0, item)
        if settle_size() > budget:
            state["context"].pop(0)
            settle_size()
    if not state.get("context"):
        state.pop("context", None)
        settle_size()
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--budget", type=int, default=8192)
    args = parser.parse_args()
    if args.budget < 128:
        parser.error("--budget must be at least 128")
    try:
        events = load_events(args.trace)
        capsule = compile_state(events)
        with open(args.trace, "rb") as stream:
            capsule["trace"] = {"events": len(events),
                                "sha256": hashlib.sha256(stream.read()).hexdigest()}
        capsule = fit_budget(capsule, args.budget)
    except (OSError, TraceError) as error:
        print(f"task-capsule: {error}", file=sys.stderr)
        return 2
    print(json.dumps(capsule, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
