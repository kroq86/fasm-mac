#!/usr/bin/env python3
"""Score a continuation answer using an explicit, non-model rubric."""

import argparse
import json
import re
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rubric")
    parser.add_argument("answer")
    args = parser.parse_args()
    try:
        rubric = json.load(open(args.rubric, encoding="utf-8"))
        answer = open(args.answer, encoding="utf-8").read().casefold()
    except (OSError, json.JSONDecodeError) as error:
        print(f"task-handoff-score: {error}", file=sys.stderr)
        return 2
    facts = rubric.get("facts", [])
    hits = [fact for fact in facts if all(term.casefold() in answer
                                          for term in fact["terms"])]
    hazards = [item for item in rubric.get("hazards", [])
               if re.search(item["pattern"], answer, re.IGNORECASE)]
    repeated = [item for item in rubric.get("repeated_work", [])
                if re.search(item["pattern"], answer, re.IGNORECASE)]
    score = len(hits) - 2 * len(hazards) - len(repeated)
    result = {"fact_hits": len(hits), "fact_total": len(facts),
              "hazards": [x["name"] for x in hazards],
              "repeated_work": [x["name"] for x in repeated], "score": score}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
