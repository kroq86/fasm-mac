#!/usr/bin/env python3
"""Natural-language variant of the lookup-binding task, checking whether the
same key->value binding result survives a less synthetic surface form.

phase1_kv_cache_sql_lookup_binding.py's sender document is a literal
"name -> id" table, and its own docstring calls this "the row store is a
KV-cache, not text" -- i.e. explicitly SQL-shaped, not natural language.
This script changes ONLY the document's surface text, from that arrow-table
to a flowing natural-language sentence per name ("Bob has ID 12345. Alice
has ID 67890. ..."), while reusing every other piece of the original
pipeline unchanged: same names (single-BPE-token, so token count per name
stays constant), same fixed-token-count id pool, same document/split
generation, same query text ("What is {name}'s ID?", already natural
language and left as-is), same adapter architecture/rank/training recipe,
same evaluation controls and verdict logic. This is a new training run on
a new (differently worded) document set, not a replacement for the
original lookup result or its replay.

If the natural-language wrapper's surrounding words tokenized differently
around different names, build_kv_batch's own `assert len(lens) == 1` would
catch it immediately (documents would fail to batch) -- this was not
independently re-verified before running; the existing assertion is the
verification.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "phase1_kv_cache_sql_lookup_binding",
    HERE / "phase1_kv_cache_sql_lookup_binding.py",
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)  # only defines functions/classes; no main() runs


def table_text_natural(order: list[str], binding: dict[str, int]) -> str:
    # No trailing punctuation directly after the id: GPT-2's BPE merges a
    # period into the preceding digits inconsistently depending on the
    # digit pattern (this was caught, not assumed -- see module docstring),
    # which broke the constant-token-length invariant build_kv_batch
    # requires for batching. A newline separator (as in the original
    # arrow-table) avoids that digit-adjacent merge entirely.
    return "\n".join(f"{n} has ID {binding[n]}" for n in order)


m.table_text = table_text_natural

if __name__ == "__main__":
    sys.exit(m.main())
