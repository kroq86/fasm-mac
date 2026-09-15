#!/usr/bin/env python3
"""Multi-DISTINCT-query amortization latency check for the lookup adapter.

Motivation: the ordering-task amortized-latency check (phase1_kv_cache_
minigunpoint_32token_amortized_latency.py) reused the SAME repeated query
against one cache, because the ordering task only has one possible query
("Class:"). The lookup task's delegation scenario is more realistic: one
sender table (e.g. name -> id bindings) can answer several DIFFERENT
downstream questions (one per name) without re-processing the table. This
script trains one lookup adapter from scratch (the lookup training script
saves no checkpoint, so there is nothing frozen to load; this mirrors
phase1_kv_cache_sql_lookup_binding_replay_result.json's same-seed retrain,
not a new architecture) and times, on ONE fixed cached table:

  (a) fixed cost: sender prefill of the table text + adapter application,
      paid once
  (b) marginal cost: one receiver query-only forward against the already-
      cached table, for a DISTINCT name each time (cycling through all
      NUM_NAMES names, not repeating one query)
  (c) marginal cost: one full receiver re-prefill of the table text plus
      that distinct query, from scratch, no sender, no adapter

CPU, single-threaded, PyTorch reference environment; not the native path.
Batch size 1 throughout (one document, one query at a time), matching the
more realistic single-request configuration already highlighted as most
relevant in the ordering-task check.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "phase1_kv_cache_sql_lookup_binding",
    HERE / "phase1_kv_cache_sql_lookup_binding.py",
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)  # only defines functions/classes; no main() runs

REPEATS = 50
WARMUP = 5


def timed(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def stats(times: list[float]) -> dict:
    s = sorted(times)
    n = len(s)
    return {"mean_ms": sum(s) / n * 1000, "median_ms": s[n // 2] * 1000,
            "min_ms": s[0] * 1000, "max_ms": s[-1] * 1000}


def train_adapter(sender, receiver, tokenizer, train_docs, dev_docs, n_layers, n_head, head_dim, device):
    train_texts = [m.table_text(d["order"], d["binding"]) for d in train_docs]
    dev_texts = [m.table_text(d["order"], d["binding"]) for d in dev_docs]
    train_k, train_v = m.build_kv_batch(sender, tokenizer, train_texts, m.SENDER_LAYER_SELECTION)
    dev_k, dev_v = m.build_kv_batch(sender, tokenizer, dev_texts, m.SENDER_LAYER_SELECTION)
    train_q, train_tgt = m.build_query_target(tokenizer, train_docs, m.target_k_cache)
    dev_q, dev_tgt = m.build_query_target(tokenizer, dev_docs, m.target_k_cache)

    adapter = m.LowRankKVAdapter(n_layers, head_dim, n_head, m.RANK, seed=m.SEED).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=m.LR)

    best_dev = math.inf
    best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
    stale = 0
    for epoch in range(m.MAX_EPOCHS):
        adapter.train()
        opt.zero_grad()
        k = adapter.forward_keys(train_k)
        v = adapter.forward_values(train_v)
        logits = m.scored_logits(receiver, train_q, train_tgt, m.make_cache(k, v), m.target_k_cache)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), train_tgt.reshape(-1))
        loss.backward()
        opt.step()
        if (epoch + 1) % m.DEV_EVAL_EVERY == 0:
            adapter.eval()
            with torch.no_grad():
                dk = adapter.forward_keys(dev_k)
                dv = adapter.forward_values(dev_v)
                dlogits = m.scored_logits(receiver, dev_q, dev_tgt, m.make_cache(dk, dv), m.target_k_cache)
                dnll, _, _ = m.per_doc_nll_top1(dlogits, dev_tgt, m.target_k_cache)
                dl = dnll.mean().item()
            adapter.train()
            if dl < best_dev - m.MIN_IMPROVEMENT:
                best_dev = dl
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= m.PATIENCE:
                break
    adapter.load_state_dict(best_state)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    return adapter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--threads", type=int, default=1, help="0 = leave PyTorch's default thread count")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = "cpu"
    m.DEVICE = device
    torch.manual_seed(m.SEED)
    random.seed(m.SEED)

    tokenizer = m.AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = m.AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval().to(device)
    receiver = m.AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval().to(device)
    for mdl in (sender, receiver):
        for p in mdl.parameters():
            p.requires_grad_(False)

    names = m.select_names(tokenizer, m.NUM_NAMES)
    target_k, id_pool = m.eligible_ids(tokenizer, m.ID_POOL_TARGET, seed=m.SEED + 1)
    m.target_k_cache = target_k  # stash for train_adapter/reuse below
    pool = m.generate_pool(names, id_pool, m.DOC_POOL_TARGET)
    train_docs = m.select_docs(pool, "train", m.TRAIN_DOCS)
    dev_docs = m.select_docs(pool, "dev", m.DEV_DOCS)
    test_docs = m.select_docs(pool, "test", m.TEST_DOCS)

    n_head = receiver.config.n_head
    head_dim = receiver.config.n_embd // n_head
    n_layers = len(m.SENDER_LAYER_SELECTION)

    print(f"training adapter (names={names})...")
    adapter = train_adapter(sender, receiver, tokenizer, train_docs, dev_docs, n_layers, n_head, head_dim, device)

    # Fix ONE cached table (first test document) as "the sender's already-processed context."
    doc = test_docs[0]
    table = m.table_text(doc["order"], doc["binding"])

    with torch.no_grad():
        def fixed_cost_path():
            k, v = m.build_kv_batch(sender, tokenizer, [table], m.SENDER_LAYER_SELECTION)
            ak = adapter.forward_keys(k)
            av = adapter.forward_values(v)
            return ak, av

        cached_k, cached_v = m.build_kv_batch(sender, tokenizer, [table], m.SENDER_LAYER_SELECTION)
        ak = adapter.forward_keys(cached_k)
        av = adapter.forward_values(cached_v)

        def marginal_query_path(name):
            cache = m.make_cache(ak, av)
            q_ids, tgt_ids = m.build_query_target(tokenizer, [doc], target_k, query_names=[name])
            return m.scored_logits(receiver, q_ids, tgt_ids, cache, target_k)

        def reprefill_path(name):
            q_text = m.query_text(name)
            tgt_text = f" {doc['binding'][name]}"
            full_text = table + "\n" + q_text
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=False)
            assert len(tgt_ids) == target_k
            ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            out = receiver(ids, use_cache=False)
            return out.logits[:, -(target_k + 1):-1, :]

        fixed_cost_path()  # sanity
        marginal_query_path(names[0])
        reprefill_path(names[0])

        fixed_times = timed(fixed_cost_path, args.repeats, WARMUP)

        marginal_times = []
        reprefill_times = []
        for i in range(args.repeats + WARMUP):
            name = names[i % len(names)]
            t0 = time.perf_counter()
            marginal_query_path(name)
            t1 = time.perf_counter()
            reprefill_path(name)
            t2 = time.perf_counter()
            if i >= WARMUP:
                marginal_times.append(t1 - t0)
                reprefill_times.append(t2 - t1)

    fixed_stats = stats(fixed_times)
    marginal_stats = stats(marginal_times)
    reprefill_stats = stats(reprefill_times)

    fixed_ms = fixed_stats["median_ms"]
    marginal_ms = marginal_stats["median_ms"]
    reprefill_ms = reprefill_stats["median_ms"]
    breakeven_n = fixed_ms / (reprefill_ms - marginal_ms) if reprefill_ms > marginal_ms else None
    sunk_speedup = reprefill_ms / marginal_ms

    result = {
        "mechanism": "lookup_multiquery_amortized_latency",
        "method": (
            "CPU, single-threaded (unless --threads 0), PyTorch reference "
            "environment. A freshly trained lookup adapter (same recipe as "
            "the same-seed replay) caches ONE table once; each timed query "
            "cycles through a DIFFERENT name from the fixed name pool "
            "(distinct downstream sub-questions against the same cached "
            "context), not a repeated identical query. Fixed cost = sender "
            "prefill of the table + adapter, paid once. Marginal cache-query "
            "= one receiver query against the cached table for one name. "
            "Marginal re-prefill = receiver alone reprocesses table+query "
            "from scratch for that name."
        ),
        "names": names,
        "repeats": args.repeats,
        "threads": args.threads if args.threads > 0 else "default",
        "fixed_cost_sender_plus_adapter": fixed_stats,
        "marginal_cost_cache_query": marginal_stats,
        "marginal_cost_reprefill": reprefill_stats,
        "breakeven_query_count": breakeven_n,
        "sunk_cost_speedup_x": sunk_speedup,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
