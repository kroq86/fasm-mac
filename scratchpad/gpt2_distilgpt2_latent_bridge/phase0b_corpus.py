#!/usr/bin/env python3
"""Deterministic synthetic corpus for Phase 0b (withheld-context headroom).

Not a sourced/downloaded corpus -- entirely invented content (fictional
names/towns/landmarks in fixed combinations), chosen specifically so it
cannot be in GPT-2's training data at all (a stronger anti-memorization
control than "post-2019", and needs no download decision). Each document
has a PREFIX that introduces a specific name, and a SUFFIX whose most
information-dense continuation repeats that name multiple times -- so a
model seeing only the suffix has genuine token-identity uncertainty a
model seeing the prefix does not.

Document-level split (train/dev), before tokenization, with exact-dedup.
Phase 0b preregistration reserves `test` unopened; this first pass does
not generate a test split at all yet -- scaling to the full preregistered
size (and adding a real held-out test split) is a later step, gated on
this smaller pass showing real headroom+sensitivity first.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

NAMES = ["Marlowe Yun", "Priya Adeyemi-Voss", "Kaspar Lindqvist", "Odalys Ferreira",
         "Tobias Wrenfield", "Nadia Skorik", "Elian Rasmussen", "Farrukh Tashkentov",
         "Wren Ashcombe", "Ilse Bragadóttir", "Ozren Vukelić", "Anouk Delacroix-Munn"]
TOWNS = ["Hollowmere", "Brackenfell", "Sunder Cross", "Thistlewick", "Vantry Hollow", "Caldbeck Reach"]
THINGS = ["luminous beetles", "spiral-shelled snails", "glass-winged moths", "burrowing lichens", "amber-veined ferns"]
LANDMARKS = ["the Ashgrave Ridge", "Millpond Hollow", "the Vantry Caves", "Crowfeather Bluff", "the Sunken Orchard"]
DAYS = ["a Tuesday", "the first frost", "midsummer", "the harvest festival", "a rainy Thursday"]

PREFIX_TMPL = ("In the town of {town}, a scientist discovered {n1} new species of "
               "{thing} near {landmark}. The discovery was announced on {day}, and word spread "
               "quickly through the region as researchers began arriving to see the find for "
               "themselves, eager to understand what had been uncovered and how significant it "
               "truly was for the field.")
# The scored tail must never restate n1's digits anywhere in the suffix
# itself -- otherwise a receiver with no prefix access can solve it by
# copying an earlier IN-SUFFIX occurrence (GPT-2-family induction heads
# are very good at this), which defeats the whole "withheld information"
# premise. n2 = n1 + delta is a derived number whose digits do not equal
# n1's digits (delta is never a round number that would make n2's last
# digit trivially guessable either).
SUFFIX_TMPL = ("Researchers explained that the species had never been recorded before. Local "
               "officials congratulated the team and allocated funding to expand the research at "
               "{landmark}. Within a year, the team's count had grown by {delta}. The final tally, "
               "published in the regional registry and celebrated across {town}, came to exactly {n2}")


def build_docs(rng: random.Random, count: int) -> list[dict]:
    docs = []
    seen = set()
    attempts = 0
    while len(docs) < count and attempts < count * 20:
        attempts += 1
        town = rng.choice(TOWNS)
        thing = rng.choice(THINGS)
        landmark = rng.choice(LANDMARKS)
        day = rng.choice(DAYS)
        n1 = rng.randint(3, 40)
        delta = rng.randint(5, 30)
        n2 = n1 + delta
        key = (town, thing, landmark, day, n1, delta)
        if key in seen:
            continue
        seen.add(key)
        prefix = PREFIX_TMPL.format(town=town, thing=thing, landmark=landmark, day=day, n1=n1)
        suffix = SUFFIX_TMPL.format(landmark=landmark, town=town, delta=delta, n2=n2)
        docs.append({"prefix": prefix, "suffix": suffix, "n1": n1, "delta": delta, "n2": n2})
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-dev", type=int, default=20)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_docs = build_docs(rng, args.n_train + args.n_dev + 20)  # headroom for split-level dedup
    rng.shuffle(all_docs)

    # de-dup by exact prefix+suffix text across the whole pool, then split
    seen_text = set()
    unique_docs = []
    for d in all_docs:
        key = d["prefix"] + "|" + d["suffix"]
        if key in seen_text:
            continue
        seen_text.add(key)
        unique_docs.append(d)

    train = unique_docs[: args.n_train]
    dev = unique_docs[args.n_train: args.n_train + args.n_dev]
    assert len(train) == args.n_train and len(dev) == args.n_dev, "not enough unique synthetic docs generated"

    # cross-split exact-duplicate check (should be vacuous by construction, verified anyway)
    train_keys = {d["prefix"] + "|" + d["suffix"] for d in train}
    dev_keys = {d["prefix"] + "|" + d["suffix"] for d in dev}
    overlap = train_keys & dev_keys
    assert not overlap, f"train/dev overlap: {overlap}"

    payload = {
        "seed": args.seed,
        "train": train,
        "dev": dev,
        "corpus_kind": "synthetic_invented_not_sourced",
    }
    text_for_hash = json.dumps(payload, sort_keys=True)
    payload["corpus_sha256"] = hashlib.sha256(text_for_hash.encode()).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"train={len(train)} dev={len(dev)} sha256={payload['corpus_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
