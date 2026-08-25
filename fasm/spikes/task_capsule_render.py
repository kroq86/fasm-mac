#!/usr/bin/env python3
"""Render a task capsule as model-neutral continuation context."""

import argparse
import json
import sys


def line(item, fields):
    body = " ".join(f"{name}={item[name]}" for name in fields if name in item)
    return f"- {body} [source:{item['source']}]"


def render(capsule):
    out = ["# Verified task continuation", ""]
    if "goal" in capsule:
        out += ["## Goal", "", line(capsule["goal"], ("text",)), ""]
    if "vcs" in capsule:
        out += ["## Repository", "", line(capsule["vcs"],
                ("branch", "head", "upstream")), ""]
    sections = (("Dirty files", "files", ("path", "state", "owned_by")),
                ("Checks", "checks", ("name", "status", "detail")),
                ("Decisions", "decisions", ("key", "text")),
                ("Next actions", "next", ("text",)))
    for title, key, fields in sections:
        if capsule.get(key):
            out += [f"## {title}", ""]
            out += [line(item, fields) for item in capsule[key]]
            out.append("")
    trace = capsule.get("trace")
    if trace:
        out += ["## Integrity", "",
                f"- trace_sha256={trace.get('sha256')} events={trace.get('events')}",
                f"- capsule_bytes={capsule.get('bytes')}", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capsule")
    args = parser.parse_args()
    try:
        with open(args.capsule, encoding="utf-8") as stream:
            print(render(json.load(stream)), end="")
    except (OSError, json.JSONDecodeError, KeyError) as error:
        print(f"task-capsule-render: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
