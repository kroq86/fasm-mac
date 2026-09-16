#!/usr/bin/env python3
"""Same 5-seed adapter-init variance check as phase1_kv_cache_sql_lookup_
binding_natural_language_seed_variance.py, but WITHOUT overriding table_text
-- i.e. the original arrow-table wording ("Bob -> 12345"). This is the
comparison point that was missing: only single points exist for the
arrow-table format (historical 20/32, replay 16/32), never a 5-seed sweep.
If the arrow-table format is ALSO fragile across seeds, the natural-language
finding is not "natural language is worse", it's "this recipe is seed-
fragile in general" -- a much bigger claim about the whole lookup line of
evidence, not just the wording check.
"""
from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "phase1_kv_cache_sql_lookup_binding",
    HERE / "phase1_kv_cache_sql_lookup_binding.py",
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)  # only defines functions/classes; no main() runs -- table_text NOT overridden

SEEDS = [20260904, 1, 2, 3, 4]


def train_and_eval(receiver, target_k, n_head, head_dim, n_layers,
                    train_k, train_v, train_q, train_tgt, dev_k, dev_v, dev_q, dev_tgt,
                    test_k, test_v, test_q, test_tgt, wrong_k, wrong_v, adapter_seed: int) -> dict:
    adapter = m.LowRankKVAdapter(n_layers, head_dim, n_head, m.RANK, seed=adapter_seed).to(m.DEVICE)
    opt = torch.optim.Adam(adapter.parameters(), lr=m.LR)

    def dev_metrics():
        adapter.eval()
        with torch.no_grad():
            k = adapter.forward_keys(dev_k)
            v = adapter.forward_values(dev_v)
            logits = m.scored_logits(receiver, dev_q, dev_tgt, m.make_cache(k, v), target_k)
            nll, exact, _ = m.per_doc_nll_top1(logits, dev_tgt, target_k)
        adapter.train()
        return nll.mean().item(), exact.mean().item()

    t0 = time.time()
    best_dev = math.inf
    best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
    stale = 0
    epochs_run = 0
    stopped_early = False
    for epoch in range(m.MAX_EPOCHS):
        adapter.train()
        opt.zero_grad()
        k = adapter.forward_keys(train_k)
        v = adapter.forward_values(train_v)
        logits = m.scored_logits(receiver, train_q, train_tgt, m.make_cache(k, v), target_k)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), train_tgt.reshape(-1))
        loss.backward()
        opt.step()
        epochs_run = epoch + 1

        if epochs_run % m.DEV_EVAL_EVERY == 0:
            dl, _ = dev_metrics()
            if dl < best_dev - m.MIN_IMPROVEMENT:
                best_dev = dl
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= m.PATIENCE:
                stopped_early = True
                break

    adapter.load_state_dict(best_state)
    adapter.eval()
    wall_s = time.time() - t0

    with torch.no_grad():
        ak = adapter.forward_keys(test_k)
        av = adapter.forward_values(test_v)
        correct_logits = m.scored_logits(receiver, test_q, test_tgt, m.make_cache(ak, av), target_k)
        correct_nll, correct_exact, _ = m.per_doc_nll_top1(correct_logits, test_tgt, target_k)

        sk, sv = m.roll_batch(test_k), m.roll_batch(test_v)
        shuf_logits = m.scored_logits(receiver, test_q, test_tgt, m.make_cache(adapter.forward_keys(sk), adapter.forward_values(sv)), target_k)
        shuf_nll, shuf_exact, _ = m.per_doc_nll_top1(shuf_logits, test_tgt, target_k)

        wk = adapter.forward_keys(wrong_k)
        wv = adapter.forward_values(wrong_v)
        wrong_logits = m.scored_logits(receiver, test_q, test_tgt, m.make_cache(wk, wv), target_k)
        wrong_nll, wrong_exact, _ = m.per_doc_nll_top1(wrong_logits, test_tgt, target_k)

    finite = all(torch.isfinite(x).all().item() for x in (correct_nll, shuf_nll, wrong_nll))
    shuffled_ci = m.delta_ci(correct_nll, shuf_nll, adapter_seed)
    wrong_binding_ci = m.delta_ci(correct_nll, wrong_nll, adapter_seed + 1)
    correct_beats_shuffled = finite and shuffled_ci["base_beats_other"]
    correct_beats_wrong_binding = finite and wrong_binding_ci["base_beats_other"]
    if correct_beats_shuffled and correct_beats_wrong_binding:
        verdict = "SQL_LOOKUP_BINDING_SUPPORTED"
    elif correct_beats_shuffled and not correct_beats_wrong_binding:
        verdict = "DOCUMENT_SPECIFIC_NO_BINDING"
    else:
        verdict = "NO_USABLE_TRANSFER"

    return {
        "adapter_seed": adapter_seed, "epochs_run": epochs_run, "stopped_early": stopped_early,
        "wall_seconds": wall_s, "exact_accuracy": correct_exact.mean().item(), "nll": correct_nll.mean().item(),
        "correct_beats_shuffled": correct_beats_shuffled, "correct_beats_wrong_binding": correct_beats_wrong_binding,
        "verdict": verdict,
    }


def main() -> int:
    import argparse
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

    names = m.select_names(tokenizer, m.NUM_NAMES)
    target_k, id_pool = m.eligible_ids(tokenizer, m.ID_POOL_TARGET, seed=m.SEED + 1)
    pool = m.generate_pool(names, id_pool, m.DOC_POOL_TARGET)
    train_docs = m.select_docs(pool, "train", m.TRAIN_DOCS)
    dev_docs = m.select_docs(pool, "dev", m.DEV_DOCS)
    test_docs = m.select_docs(pool, "test", m.TEST_DOCS)

    n_head = receiver.config.n_head
    head_dim = receiver.config.n_embd // n_head
    n_layers = len(m.SENDER_LAYER_SELECTION)

    train_texts = [m.table_text(d["order"], d["binding"]) for d in train_docs]
    dev_texts = [m.table_text(d["order"], d["binding"]) for d in dev_docs]
    test_texts = [m.table_text(d["order"], d["binding"]) for d in test_docs]
    train_k, train_v = m.build_kv_batch(sender, tokenizer, train_texts, m.SENDER_LAYER_SELECTION)
    dev_k, dev_v = m.build_kv_batch(sender, tokenizer, dev_texts, m.SENDER_LAYER_SELECTION)
    test_k, test_v = m.build_kv_batch(sender, tokenizer, test_texts, m.SENDER_LAYER_SELECTION)
    train_q, train_tgt = m.build_query_target(tokenizer, train_docs, target_k)
    dev_q, dev_tgt = m.build_query_target(tokenizer, dev_docs, target_k)
    test_q, test_tgt = m.build_query_target(tokenizer, test_docs, target_k)

    wrong_docs = [m.wrong_binding_doc(d, names) for d in test_docs]
    wrong_texts = [m.table_text(d["order"], d["binding"]) for d in wrong_docs]
    wrong_k, wrong_v = m.build_kv_batch(sender, tokenizer, wrong_texts, m.SENDER_LAYER_SELECTION)

    runs = []
    for seed in SEEDS:
        r = train_and_eval(receiver, target_k, n_head, head_dim, n_layers,
                            train_k, train_v, train_q, train_tgt, dev_k, dev_v, dev_q, dev_tgt,
                            test_k, test_v, test_q, test_tgt, wrong_k, wrong_v, seed)
        print(f"adapter_seed={seed} exact_accuracy={r['exact_accuracy']:.4f} nll={r['nll']:.4f} verdict={r['verdict']} wall_s={r['wall_seconds']:.1f}")
        runs.append(r)

    accuracies = [r["exact_accuracy"] for r in runs]
    result = {
        "mechanism": "training_seed_variance",
        "method": (
            "Same fixed train/dev/test split and original arrow-table "
            "document wording ('name -> id', unmodified table_text) as the "
            "historical and replayed lookup runs; only the LowRankKVAdapter's "
            "random initialization seed varies across runs, each trained to "
            "the same early-stopping budget with identical hyperparameters. "
            "Same protocol as lookup_natural_language_seed_variance, applied "
            "to the original wording instead, so the two are a controlled "
            "comparison of surface form at matched seeds. The first seed "
            "(20260904) matches phase1_kv_cache_sql_lookup_binding_replay_"
            "result.json's same-seed retrain (both give 16/32)."
        ),
        "seeds": SEEDS, "runs": runs,
        "accuracy_mean": sum(accuracies) / len(accuracies),
        "accuracy_min": min(accuracies), "accuracy_max": max(accuracies),
        "accuracy_range_in_32ths": [round(a * 32) for a in accuracies],
        "verdicts": [r["verdict"] for r in runs],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
