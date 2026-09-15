#!/usr/bin/env python3
"""Training-seed variance check for the ordering (mini-GunPoint) adapter.

Motivation: README's "Remaining submission decisions" names "seed variance"
as missing validation. The historical ordering result trains one adapter at
one fixed seed (20260904) for 800 epochs and reports 31/32 dev accuracy.
This script holds the dataset fixed (train/dev split comes from
generate_pool/select_docs, which depend only on the module's SEED constant,
left untouched here) and varies only the LowRankKVAdapter's random
initialization across several seeds, training each to the same 800-epoch
budget with the same hyperparameters, to see how much final dev accuracy
varies with initialization alone. This is a new set of training runs, not
inference over the frozen checkpoint; the original checkpoint and its
result file are untouched.

Reuses generate_pool/select_docs/build_batch/LowRankKVAdapter/etc. from the
original training script as a module (not duplicated); only the adapter
init seed and the training loop invocation are written here, since the
original script's main() hardcodes the same SEED for data, adapter, and
torch's global RNG (which the adapter init does not actually consume, since
it uses its own torch.Generator).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "phase1_kv_cache_minigunpoint_32token",
    HERE / "phase1_kv_cache_minigunpoint_32token.py",
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)  # only defines functions/classes; no main() runs

SEEDS = [20260904, 1, 2, 3, 4]  # first is the historical adapter-init seed


def train_one(sender, receiver, tokenizer, n_head, head_dim, n_layers,
              train_k, train_v, train_target, dev_k, dev_v, dev_target,
              tq, dq, adapter_seed: int) -> dict:
    adapter = m.LowRankKVAdapter(n_layers, head_dim, n_head, m.RANK, seed=adapter_seed).to(m.DEVICE)
    opt = torch.optim.Adam(adapter.parameters(), lr=m.LR)

    best_dev = math.inf
    best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
    t0 = time.time()
    for epoch in range(m.MAX_EPOCHS):
        adapter.train()
        opt.zero_grad()
        k = adapter.forward_keys(train_k)
        v = adapter.forward_values(train_v)
        logits = m.next_token_logits(receiver, tq, m.make_cache(k, v))
        loss = torch.nn.functional.cross_entropy(logits, train_target)
        loss.backward()
        opt.step()

        if (epoch + 1) % m.DEV_EVAL_EVERY == 0 or epoch == m.MAX_EPOCHS - 1:
            adapter.eval()
            with torch.no_grad():
                dk = adapter.forward_keys(dev_k)
                dv = adapter.forward_values(dev_v)
                dlogits = m.next_token_logits(receiver, dq, m.make_cache(dk, dv))
                dloss = torch.nn.functional.cross_entropy(dlogits, dev_target).item()
            adapter.train()
            if dloss < best_dev:
                best_dev = dloss
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
    wall_s = time.time() - t0

    adapter.load_state_dict(best_state)
    adapter.eval()
    with torch.no_grad():
        correct_logits = m.next_token_logits(receiver, dq, m.make_cache(adapter.forward_keys(dev_k), adapter.forward_values(dev_v)))
        correct_nll, correct_top1 = m.nll_top1(correct_logits, dev_target)
    return {
        "adapter_seed": adapter_seed,
        "wall_seconds": wall_s,
        "best_dev_loss_at_selection": best_dev,
        "accuracy": correct_top1.mean().item(),
        "nll": correct_nll.mean().item(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    tokenizer = m.AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = m.AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval().to(m.DEVICE)
    receiver = m.AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval().to(m.DEVICE)
    for mod in (sender, receiver):
        for p in mod.parameters():
            p.requires_grad_(False)

    pool = m.generate_pool(target_per_class=200)
    train_docs = m.select_docs(pool, "train", m.TRAIN_DOCS)
    dev_docs = m.select_docs(pool, "dev", m.DEV_DOCS)
    train_keys_set = {"".join(s) for s, _ in train_docs}
    dev_keys_set = {"".join(s) for s, _ in dev_docs}
    assert train_keys_set.isdisjoint(dev_keys_set)

    train_k, train_v, train_target = m.build_batch(sender, tokenizer, train_docs)
    dev_k, dev_v, dev_target = m.build_batch(sender, tokenizer, dev_docs)
    n_head = receiver.config.n_head
    head_dim = receiver.config.n_embd // n_head
    n_layers = len(m.SENDER_LAYER_SELECTION)
    tq = m.query_ids(tokenizer, len(train_docs))
    dq = m.query_ids(tokenizer, len(dev_docs))

    runs = []
    for seed in SEEDS:
        r = train_one(sender, receiver, tokenizer, n_head, head_dim, n_layers,
                       train_k, train_v, train_target, dev_k, dev_v, dev_target,
                       tq, dq, seed)
        print(f"adapter_seed={seed} acc={r['accuracy']:.4f} nll={r['nll']:.4f} wall_s={r['wall_seconds']:.1f}")
        runs.append(r)

    accuracies = [r["accuracy"] for r in runs]
    result = {
        "mechanism": "training_seed_variance",
        "method": (
            "Same fixed train/dev split (generate_pool/select_docs at the "
            "module's original SEED, untouched); only the LowRankKVAdapter's "
            "random initialization seed varies across runs, each trained for "
            "the full 800-epoch budget with identical hyperparameters. The "
            "first seed (20260904) matches the historical adapter-init seed."
        ),
        "seeds": SEEDS,
        "runs": runs,
        "accuracy_mean": sum(accuracies) / len(accuracies),
        "accuracy_min": min(accuracies),
        "accuracy_max": max(accuracies),
        "accuracy_range_in_32ths": [round(a * 32) for a in accuracies],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
