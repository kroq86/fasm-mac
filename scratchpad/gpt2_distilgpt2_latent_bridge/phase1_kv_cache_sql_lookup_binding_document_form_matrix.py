#!/usr/bin/env python3
"""Document-form x seed matrix for the lookup-binding task, all at the
tuned LR=1e-3 established by phase1_kv_cache_sql_lookup_binding_natural_
language_lr_sweep_result.json (untuned 1e-2 collapses training on longer/
noisier forms and makes form comparisons meaningless -- see also
phase1_kv_cache_sql_lookup_binding_natural_language_rank_sweep_result.json,
which ruled out adapter rank as the bottleneck first).

Tests a token-count gradient of document-row separators between the
arrow-table format ("Bob -> 12345") and the natural-language sentence
format ("Bob has ID 12345"): colon and "is" keep the SAME per-line token
count as arrow (verified against the tokenizer before running, per-line
lengths [4,4,5,5,4] for all three); has_id adds 5 tokens total, id_is adds
10. This isolates whether the natural-language format's failure (Section
5.2) is about "naturalness" of the wording or about token distance between
key and value in the cache, by testing a wording that is fully natural
English ("is") at the arrow-table's own token length.
"""
from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("phase1_kv_cache_sql_lookup_binding", HERE / "phase1_kv_cache_sql_lookup_binding.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

TUNED_LR = 1e-3
SEEDS = [20260904, 1, 2, 3, 4]

FORMS = {
    "arrow":  lambda n, v: f"{n} -> {v}",
    "colon":  lambda n, v: f"{n}: {v}",
    "is":     lambda n, v: f"{n} is {v}",
    "has_id": lambda n, v: f"{n} has ID {v}",
    "id_is":  lambda n, v: f"{n}'s ID is {v}",
}
# "equals" (f"{n}={v}") was tried and dropped: BPE merges "=" with certain
# digit patterns inconsistently, breaking the constant-token-length
# invariant build_kv_batch requires across the full id pool (caught, not
# assumed: AssertionError on the dev split, {26, 27} tokens).


def train_and_eval(sender, receiver, target_k, n_head, head_dim, n_layers,
                    train_k, train_v, train_q, train_tgt, dev_k, dev_v, dev_q, dev_tgt,
                    test_k, test_v, test_q, test_tgt, wrong_k, wrong_v, adapter_seed: int) -> dict:
    adapter = m.LowRankKVAdapter(n_layers, head_dim, n_head, m.RANK, seed=adapter_seed).to(m.DEVICE)
    opt = torch.optim.Adam(adapter.parameters(), lr=TUNED_LR)

    def dev_metrics():
        adapter.eval()
        with torch.no_grad():
            k = adapter.forward_keys(dev_k); v = adapter.forward_values(dev_v)
            logits = m.scored_logits(receiver, dev_q, dev_tgt, m.make_cache(k, v), target_k)
            nll, exact, _ = m.per_doc_nll_top1(logits, dev_tgt, target_k)
        adapter.train()
        return nll.mean().item()

    t0 = time.time()
    best_dev = math.inf
    best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
    stale = 0
    epochs_run = 0
    stopped_early = False
    for epoch in range(m.MAX_EPOCHS):
        adapter.train()
        opt.zero_grad()
        k = adapter.forward_keys(train_k); v = adapter.forward_values(train_v)
        logits = m.scored_logits(receiver, train_q, train_tgt, m.make_cache(k, v), target_k)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), train_tgt.reshape(-1))
        loss.backward(); opt.step()
        epochs_run = epoch + 1
        if epochs_run % m.DEV_EVAL_EVERY == 0:
            dl = dev_metrics()
            if dl < best_dev - m.MIN_IMPROVEMENT:
                best_dev = dl; best_state = {k: v.clone() for k, v in adapter.state_dict().items()}; stale = 0
            else:
                stale += 1
            if stale >= m.PATIENCE:
                stopped_early = True; break

    adapter.load_state_dict(best_state)
    adapter.eval()
    wall_s = time.time() - t0

    with torch.no_grad():
        ak = adapter.forward_keys(test_k); av = adapter.forward_values(test_v)
        correct_logits = m.scored_logits(receiver, test_q, test_tgt, m.make_cache(ak, av), target_k)
        correct_nll, correct_exact, _ = m.per_doc_nll_top1(correct_logits, test_tgt, target_k)
        sk, sv = m.roll_batch(test_k), m.roll_batch(test_v)
        shuf_logits = m.scored_logits(receiver, test_q, test_tgt, m.make_cache(adapter.forward_keys(sk), adapter.forward_values(sv)), target_k)
        shuf_nll, shuf_exact, _ = m.per_doc_nll_top1(shuf_logits, test_tgt, target_k)
        wk = adapter.forward_keys(wrong_k); wv = adapter.forward_values(wrong_v)
        wrong_logits = m.scored_logits(receiver, test_q, test_tgt, m.make_cache(wk, wv), target_k)
        wrong_nll, wrong_exact, _ = m.per_doc_nll_top1(wrong_logits, test_tgt, target_k)

    finite = all(torch.isfinite(x).all().item() for x in (correct_nll, shuf_nll, wrong_nll))
    shuffled_ci = m.delta_ci(correct_nll, shuf_nll, adapter_seed)
    wrong_binding_ci = m.delta_ci(correct_nll, wrong_nll, adapter_seed + 1)
    correct_beats_shuffled = finite and shuffled_ci["base_beats_other"]
    correct_beats_wrong_binding = finite and wrong_binding_ci["base_beats_other"]
    if correct_beats_shuffled and correct_beats_wrong_binding:
        verdict = "SUPPORTED"
    elif correct_beats_shuffled:
        verdict = "DOC_SPECIFIC"
    else:
        verdict = "NO_TRANSFER"
    return {"adapter_seed": adapter_seed, "exact_accuracy": correct_exact.mean().item(),
            "nll": correct_nll.mean().item(), "verdict": verdict, "epochs_run": epochs_run, "wall_s": wall_s}


def main():
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

    all_results = {}
    for form_name, template in FORMS.items():
        def table_text_form(order, binding, template=template):
            return "\n".join(template(n, binding[n]) for n in order)

        train_texts = [table_text_form(d["order"], d["binding"]) for d in train_docs]
        dev_texts = [table_text_form(d["order"], d["binding"]) for d in dev_docs]
        test_texts = [table_text_form(d["order"], d["binding"]) for d in test_docs]
        train_k, train_v = m.build_kv_batch(sender, tokenizer, train_texts, m.SENDER_LAYER_SELECTION)
        dev_k, dev_v = m.build_kv_batch(sender, tokenizer, dev_texts, m.SENDER_LAYER_SELECTION)
        test_k, test_v = m.build_kv_batch(sender, tokenizer, test_texts, m.SENDER_LAYER_SELECTION)
        train_q, train_tgt = m.build_query_target(tokenizer, train_docs, target_k)
        dev_q, dev_tgt = m.build_query_target(tokenizer, dev_docs, target_k)
        test_q, test_tgt = m.build_query_target(tokenizer, test_docs, target_k)
        wrong_docs = [m.wrong_binding_doc(d, names) for d in test_docs]
        wrong_texts = [table_text_form(d["order"], d["binding"]) for d in wrong_docs]
        wrong_k, wrong_v = m.build_kv_batch(sender, tokenizer, wrong_texts, m.SENDER_LAYER_SELECTION)

        runs = []
        for seed in SEEDS:
            r = train_and_eval(sender, receiver, target_k, n_head, head_dim, n_layers,
                                train_k, train_v, train_q, train_tgt, dev_k, dev_v, dev_q, dev_tgt,
                                test_k, test_v, test_q, test_tgt, wrong_k, wrong_v, seed)
            print(f"form={form_name:8s} seed={seed:>10} acc={r['exact_accuracy']:.4f} verdict={r['verdict']} wall_s={r['wall_s']:.1f}")
            runs.append(r)
        accs = [r["exact_accuracy"] for r in runs]
        supported = sum(1 for r in runs if r["verdict"] == "SUPPORTED")
        all_results[form_name] = {"runs": runs, "accuracy_mean": sum(accs) / len(accs),
                                   "supported_count": supported, "accuracy_range_in_32ths": [round(a * 32) for a in accs]}

    print("\n=== MATRIX SUMMARY (tuned LR=1e-3, 5 seeds each) ===")
    for form_name, res in all_results.items():
        print(f"{form_name:8s} supported={res['supported_count']}/5  mean_acc_32={res['accuracy_mean']*32:.1f}  range={res['accuracy_range_in_32ths']}")

    result = {
        "mechanism": "document_form_matrix",
        "method": (
            "Same fixed train/dev/test split, architecture, and tuned "
            "learning rate (1e-3) as lookup_natural_language_tuned_lr_"
            "seed_variance; only the per-name-per-row document template "
            "changes across five forms, each checked across the same five "
            "adapter-init seeds. 'equals' (f'{name}={id}') was tried and "
            "dropped: BPE merges '=' with certain digit patterns "
            "inconsistently, breaking the constant-token-length invariant "
            "build_kv_batch requires across the full id pool."
        ),
        "tuned_lr": TUNED_LR, "seeds": SEEDS,
        "form_token_lengths": {"arrow": 26, "colon": 26, "is": 26, "has_id": 31, "id_is": 36},
        "forms": all_results,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
