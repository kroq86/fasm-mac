#!/usr/bin/env python3
"""SQL-style key-value lookup via KV-cache handoff: the cleanest available
addressing+binding test. Sender sees a table of NUM_NAMES (name -> id)
pairs and builds a KV-cache; the receiver sees ONLY that transferred cache
plus a query naming ONE person, and must answer with that person's id --
"SELECT id WHERE name = ?", except the row store is a KV-cache, not text.

Names are drawn from a FIXED, tokenizer-verified single-BPE-token pool of
NUM_NAMES names (so the printed table's total token length, and the
query's total token length, stay constant across documents regardless of
which name is queried -- required for batched training/eval). Per
document: name DISPLAY ORDER is reshuffled (so position cannot leak
identity), and the name->id BINDING (a bijection over the fixed name set)
is freshly randomized with unique ids. The query asks for one of the 5
names, chosen uniformly at random per document.

Controls (per document, at eval time):
  - correct:          the document's own adapted KV.
  - shuffled_document: a DIFFERENT document's adapted KV (same name pool,
    different ids/binding/order) -- tests whether the cache carries
    document-specific content at all.
  - wrong_binding:     the SAME set of id VALUES, reassigned to names by a
    fixed rotation (a derangement -- no name keeps its true id), rendered
    as its own fresh table text and passed through the frozen sender
    again -- same display order preserved, only which id follows which
    name changes. Tests binding specifically, holding the value set fixed.
  - zero_sender:       zero-valued KV of the same shape.
  - query_name_swap (secondary, not part of the primary gate): the
    document's own correct KV, but scored against a DIFFERENT name's
    (still true, same document) id -- a positive sanity control: if
    query-conditioning works at all, this should also succeed.

Primary gate: correct must beat BOTH shuffled_document AND wrong_binding
on a paired per-document bootstrap CI, evaluated ONCE on a test split that
is not touched (not even tokenized) until the training/early-stopping
protocol below is complete.

Interpretation (fixed before running, not chosen after seeing results):
  - correct ~= zero:                          NO_USABLE_TRANSFER
  - correct > shuffled, correct ~= wrong_binding:
        DOCUMENT_SPECIFIC_NO_BINDING (value-set/global signal only)
  - correct > shuffled AND correct > wrong_binding:
        SQL_LOOKUP_BINDING_SUPPORTED
  - anything else (e.g. still improving at the epoch cap):
        INCONCLUSIVE

Recipe choices are those already validated earlier this session: full
batch (one forward/backward/opt.step per epoch over the whole train set,
not a per-example loop), MPS device, low-rank shared KV adapter (same
class/rank/layer-selection as the mini-GunPoint bridge), LR=1e-2 (the
value the same-session LR diagnostic identified as the strongest of three
candidates once full-batch training was used). This run is the sole
process on the machine; nothing else runs in parallel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

SEED = 20260904
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

NUM_NAMES = 5
ID_MIN, ID_MAX = 10000, 99999
ID_POOL_TARGET = 300
DOC_POOL_TARGET = 400

TRAIN_DOCS = 64
DEV_DOCS = 32
TEST_DOCS = 32

SENDER_LAYER_SELECTION = (1, 3, 5, 7, 9, 11)
RANK = 8
A_INIT_STD = 0.02

LR = 1e-2
MAX_EPOCHS = 800
DEV_EVAL_EVERY = 5
PATIENCE = 10  # in dev-eval units -> 50 epochs without improvement
MIN_IMPROVEMENT = 1e-4

NAME_CANDIDATES = [
    "Bob", "Alice", "Ivan", "Lena", "Anna", "Nina", "Peter", "Boris",
    "Masha", "Igor", "Olga", "Kate", "Tom", "Max", "Eva", "Sam", "Jack",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bucket(key: str) -> str:
    value = int(sha(key.encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def select_names(tokenizer, k: int) -> list[str]:
    chosen = []
    for name in NAME_CANDIDATES:
        if len(tokenizer.encode(f" {name}", add_special_tokens=False)) == 1:
            chosen.append(name)
        if len(chosen) == k:
            break
    if len(chosen) < k:
        raise RuntimeError(f"only found {len(chosen)} single-BPE-token name candidates, need {k}")
    return chosen


def eligible_ids(tokenizer, pool_target: int, seed: int) -> tuple[int, list[int]]:
    rng = random.Random(seed)
    seen: set[int] = set()
    buckets: dict[int, list[int]] = {}
    while True:
        v = rng.randint(ID_MIN, ID_MAX)
        if v in seen:
            continue
        seen.add(v)
        n = len(tokenizer.encode(f" {v}", add_special_tokens=False))
        buckets.setdefault(n, []).append(v)
        if len(buckets[n]) >= pool_target:
            return n, buckets[n]
        if len(seen) > 300000:
            raise RuntimeError("could not find enough fixed-token-count ids")


def query_text(name: str) -> str:
    return f"What is {name}'s ID?"


def table_text(order: list[str], binding: dict[str, int]) -> str:
    return "\n".join(f"{n} -> {binding[n]}" for n in order)


def doc_key(order: list[str], binding: dict[str, int], names: list[str], query_name: str) -> str:
    parts = [f"{n}:{binding[n]}" for n in names]
    return "|".join(parts) + f"|order:{','.join(order)}|q:{query_name}"


def make_doc(rng: random.Random, names: list[str], id_pool: list[int]) -> dict:
    order = names[:]
    rng.shuffle(order)
    ids = rng.sample(id_pool, NUM_NAMES)
    binding = dict(zip(names, ids))  # canonical NAMES-order bijection
    query_name = rng.choice(names)
    return {"order": order, "binding": binding, "query_name": query_name}


def wrong_binding_doc(doc: dict, names: list[str]) -> dict:
    ids_in_name_order = [doc["binding"][n] for n in names]
    rotated = ids_in_name_order[1:] + ids_in_name_order[:1]
    wrong = dict(zip(names, rotated))
    assert all(wrong[n] != doc["binding"][n] for n in names), "rotation must be a full derangement"
    return {"order": doc["order"], "binding": wrong, "query_name": doc["query_name"]}


def generate_pool(names: list[str], id_pool: list[int], target: int) -> list[dict]:
    rng = random.Random(SEED)
    pool, seen = [], set()
    attempts = 0
    while len(pool) < target and attempts < target * 50:
        attempts += 1
        d = make_doc(rng, names, id_pool)
        k = doc_key(d["order"], d["binding"], names, d["query_name"])
        if k in seen:
            continue
        seen.add(k)
        d["_key"] = k
        pool.append(d)
    if len(pool) < target:
        raise RuntimeError(f"could not generate {target} unique documents")
    return pool


def select_docs(pool: list[dict], split: str, count: int) -> list[dict]:
    chosen = sorted((d for d in pool if bucket(d["_key"]) == split), key=lambda d: d["_key"])
    if len(chosen) < count:
        raise RuntimeError(f"only {len(chosen)} eligible {split} docs")
    return chosen[:count]


def build_kv_batch(sender, tokenizer, texts: list[str], layer_sel):
    tokenized = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    lens = {len(t) for t in tokenized}
    assert len(lens) == 1, f"table length not constant: {lens}"
    ids = torch.tensor(tokenized, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = sender(ids, use_cache=True)
    pkv = out.past_key_values
    keys = [pkv.layers[i].keys.clone().detach() for i in layer_sel]
    values = [pkv.layers[i].values.clone().detach() for i in layer_sel]
    return keys, values


def build_query_target(tokenizer, docs: list[dict], target_k: int, query_names: list[str] | None = None):
    qn = query_names if query_names is not None else [d["query_name"] for d in docs]
    q_tok = [tokenizer.encode(query_text(n), add_special_tokens=False) for n in qn]
    qlens = {len(q) for q in q_tok}
    assert len(qlens) == 1, f"query length not constant: {qlens}"
    q_ids = torch.tensor(q_tok, dtype=torch.long, device=DEVICE)
    tgt_tok = []
    for d, n in zip(docs, qn):
        t = tokenizer.encode(f" {d['binding'][n]}", add_special_tokens=False)
        assert len(t) == target_k
        tgt_tok.append(t)
    tgt_ids = torch.tensor(tgt_tok, dtype=torch.long, device=DEVICE)
    return q_ids, tgt_ids


class LowRankKVAdapter(nn.Module):
    def __init__(self, n_layers: int, head_dim: int, n_head: int, rank: int, seed: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        width = n_head * head_dim
        g = torch.Generator().manual_seed(seed)
        self.A_k = nn.ParameterList([nn.Parameter(torch.randn(width, rank, generator=g) * A_INIT_STD) for _ in range(n_layers)])
        self.B_k = nn.ParameterList([nn.Parameter(torch.zeros(rank, width)) for _ in range(n_layers)])
        self.A_v = nn.ParameterList([nn.Parameter(torch.randn(width, rank, generator=g) * A_INIT_STD) for _ in range(n_layers)])
        self.B_v = nn.ParameterList([nn.Parameter(torch.zeros(rank, width)) for _ in range(n_layers)])

    def _lowrank_correct(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(b * s, h * d)
        correction = (flat @ A) @ B
        out_flat = flat + correction
        return out_flat.reshape(b, s, h, d).permute(0, 2, 1, 3)

    def forward_keys(self, keys):
        return [self._lowrank_correct(k, self.A_k[i], self.B_k[i]) for i, k in enumerate(keys)]

    def forward_values(self, values):
        return [self._lowrank_correct(v, self.A_v[i], self.B_v[i]) for i, v in enumerate(values)]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_cache(keys, values) -> DynamicCache:
    cache = DynamicCache()
    for k, v in zip(keys, values):
        cache.update(k, v, layer_idx=len(cache.layers))
    return cache


def zero_like(keys):
    return [torch.zeros_like(k) for k in keys]


def roll_batch(keys):
    return [torch.roll(k, shifts=1, dims=0) for k in keys]


def scored_logits(receiver, q_ids: torch.Tensor, tgt_ids: torch.Tensor, cache: DynamicCache, k: int) -> torch.Tensor:
    receiver_ids = torch.cat([q_ids, tgt_ids], dim=1)
    out = receiver(receiver_ids, past_key_values=cache, use_cache=False)
    return out.logits[:, -(k + 1):-1, :].float()


def per_doc_nll_top1(logits: torch.Tensor, target: torch.Tensor, k: int):
    n = logits.shape[0]
    per_tok = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
    ).reshape(n, k)
    nll = per_tok.mean(dim=1)
    exact = (logits.argmax(-1) == target).all(dim=1).float()
    per_tok_top1 = (logits.argmax(-1) == target).float().mean(dim=0)  # (k,)
    return nll, exact, per_tok_top1


def bootstrap_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return [means[249], means[9749]]


def summarize(nll: torch.Tensor, exact: torch.Tensor, per_tok_top1: torch.Tensor) -> dict:
    return {
        "nll": nll.mean().item(), "exact_accuracy": exact.mean().item(),
        "per_token_top1": per_tok_top1.tolist(),
    }


def delta_ci(base_nll: torch.Tensor, other_nll: torch.Tensor, seed: int) -> dict:
    delta = (other_nll - base_nll).tolist()
    ci = bootstrap_ci(delta, seed)
    return {"other_minus_base_nll": sum(delta) / len(delta), "bootstrap_95pct_ci": ci, "base_beats_other": ci[0] > 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval().to(DEVICE)
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval().to(DEVICE)
    for m in (sender, receiver):
        for p in m.parameters():
            p.requires_grad_(False)

    names = select_names(tokenizer, NUM_NAMES)
    target_k, id_pool = eligible_ids(tokenizer, ID_POOL_TARGET, seed=SEED + 1)
    pool = generate_pool(names, id_pool, DOC_POOL_TARGET)

    train_docs = select_docs(pool, "train", TRAIN_DOCS)
    dev_docs = select_docs(pool, "dev", DEV_DOCS)
    test_docs = select_docs(pool, "test", TEST_DOCS)
    all_keys = {d["_key"] for d in train_docs} | {d["_key"] for d in dev_docs} | {d["_key"] for d in test_docs}
    assert len(all_keys) == TRAIN_DOCS + DEV_DOCS + TEST_DOCS, "split leakage across train/dev/test"

    print(f"names={names} target_id_tokens={target_k} id_pool={len(id_pool)}")

    n_head = receiver.config.n_head
    head_dim = receiver.config.n_embd // n_head
    n_layers = len(SENDER_LAYER_SELECTION)

    train_texts = [table_text(d["order"], d["binding"]) for d in train_docs]
    dev_texts = [table_text(d["order"], d["binding"]) for d in dev_docs]
    train_k, train_v = build_kv_batch(sender, tokenizer, train_texts, SENDER_LAYER_SELECTION)
    dev_k, dev_v = build_kv_batch(sender, tokenizer, dev_texts, SENDER_LAYER_SELECTION)
    train_q, train_tgt = build_query_target(tokenizer, train_docs, target_k)
    dev_q, dev_tgt = build_query_target(tokenizer, dev_docs, target_k)

    adapter = LowRankKVAdapter(n_layers, head_dim, n_head, RANK, seed=SEED).to(DEVICE)
    opt = torch.optim.Adam(adapter.parameters(), lr=LR)
    print(f"trainable_params={adapter.param_count()}")

    k0 = adapter.forward_keys(train_k)
    v0 = adapter.forward_values(train_v)
    logits0 = scored_logits(receiver, train_q, train_tgt, make_cache(k0, v0), target_k)
    loss0 = torch.nn.functional.cross_entropy(logits0.reshape(-1, logits0.shape[-1]), train_tgt.reshape(-1))
    loss0.backward()
    grad_nonzero = any(adapter.B_k[i].grad.abs().max().item() > 0 for i in range(n_layers)) or \
                   any(adapter.B_v[i].grad.abs().max().item() > 0 for i in range(n_layers))
    if not grad_nonzero:
        raise RuntimeError("SQL-lookup KV adapter has no learning signal at init")
    opt.zero_grad()

    t0 = time.time()
    best_dev = math.inf
    best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
    stale = 0
    train_loss_history: list[float] = []
    dev_eval_epochs: list[int] = []
    dev_loss_history: list[float] = []
    dev_acc_history: list[float] = []
    stopped_early = False

    def dev_metrics():
        adapter.eval()
        with torch.no_grad():
            k = adapter.forward_keys(dev_k)
            v = adapter.forward_values(dev_v)
            logits = scored_logits(receiver, dev_q, dev_tgt, make_cache(k, v), target_k)
            nll, exact, _ = per_doc_nll_top1(logits, dev_tgt, target_k)
        adapter.train()
        return nll.mean().item(), exact.mean().item()

    for epoch in range(MAX_EPOCHS):
        adapter.train()
        opt.zero_grad()
        k = adapter.forward_keys(train_k)
        v = adapter.forward_values(train_v)
        logits = scored_logits(receiver, train_q, train_tgt, make_cache(k, v), target_k)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), train_tgt.reshape(-1))
        loss.backward()
        opt.step()
        train_loss_history.append(loss.item())

        if (epoch + 1) % DEV_EVAL_EVERY == 0:
            dl, dacc = dev_metrics()
            dev_eval_epochs.append(epoch + 1)
            dev_loss_history.append(dl)
            dev_acc_history.append(dacc)
            if dl < best_dev - MIN_IMPROVEMENT:
                best_dev = dl
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= PATIENCE:
                stopped_early = True
                break

    epochs_capped = not stopped_early
    adapter.load_state_dict(best_state)
    adapter.eval()
    wall_s = time.time() - t0
    print(f"training done: epochs_run={len(train_loss_history)} stopped_early={stopped_early} "
          f"epochs_capped={epochs_capped} best_dev_loss={best_dev:.4f} wall_s={wall_s:.1f}")

    # ---- test split opens here, single pass, all arms ----
    test_texts = [table_text(d["order"], d["binding"]) for d in test_docs]
    test_k, test_v = build_kv_batch(sender, tokenizer, test_texts, SENDER_LAYER_SELECTION)
    test_q, test_tgt = build_query_target(tokenizer, test_docs, target_k)
    n_test = len(test_docs)

    wrong_docs = [wrong_binding_doc(d, names) for d in test_docs]
    wrong_texts = [table_text(d["order"], d["binding"]) for d in wrong_docs]
    wrong_k, wrong_v = build_kv_batch(sender, tokenizer, wrong_texts, SENDER_LAYER_SELECTION)

    swap_names = [names[(names.index(d["query_name"]) + 1) % NUM_NAMES] for d in test_docs]
    swap_q, swap_tgt = build_query_target(tokenizer, test_docs, target_k, query_names=swap_names)

    with torch.no_grad():
        ak = adapter.forward_keys(test_k)
        av = adapter.forward_values(test_v)
        correct_logits = scored_logits(receiver, test_q, test_tgt, make_cache(ak, av), target_k)
        correct_nll, correct_exact, correct_top1 = per_doc_nll_top1(correct_logits, test_tgt, target_k)

        sk, sv = roll_batch(test_k), roll_batch(test_v)
        shuf_logits = scored_logits(receiver, test_q, test_tgt, make_cache(adapter.forward_keys(sk), adapter.forward_values(sv)), target_k)
        shuf_nll, shuf_exact, shuf_top1 = per_doc_nll_top1(shuf_logits, test_tgt, target_k)

        wk = adapter.forward_keys(wrong_k)
        wv = adapter.forward_values(wrong_v)
        wrong_logits = scored_logits(receiver, test_q, test_tgt, make_cache(wk, wv), target_k)
        wrong_nll, wrong_exact, wrong_top1 = per_doc_nll_top1(wrong_logits, test_tgt, target_k)

        zk, zv = zero_like(test_k), zero_like(test_v)
        zero_logits = scored_logits(receiver, test_q, test_tgt, make_cache(adapter.forward_keys(zk), adapter.forward_values(zv)), target_k)
        zero_nll, zero_exact, zero_top1 = per_doc_nll_top1(zero_logits, test_tgt, target_k)

        swap_logits = scored_logits(receiver, swap_q, swap_tgt, make_cache(ak, av), target_k)
        swap_nll, swap_exact, swap_top1 = per_doc_nll_top1(swap_logits, swap_tgt, target_k)

    finite = all(torch.isfinite(x).all().item() for x in (correct_nll, shuf_nll, wrong_nll, zero_nll, swap_nll))

    shuffled_ci = delta_ci(correct_nll, shuf_nll, SEED)
    wrong_binding_ci = delta_ci(correct_nll, wrong_nll, SEED + 1)
    zero_ci = delta_ci(correct_nll, zero_nll, SEED + 2)

    correct_beats_shuffled = finite and shuffled_ci["base_beats_other"]
    correct_beats_wrong_binding = finite and wrong_binding_ci["base_beats_other"]
    correct_beats_zero = finite and zero_ci["base_beats_other"]

    if epochs_capped:
        verdict = "INCONCLUSIVE_EPOCHS_CAPPED"
    elif not correct_beats_zero:
        verdict = "NO_USABLE_TRANSFER"
    elif correct_beats_shuffled and correct_beats_wrong_binding:
        verdict = "SQL_LOOKUP_BINDING_SUPPORTED"
    elif correct_beats_shuffled and not correct_beats_wrong_binding:
        verdict = "DOCUMENT_SPECIFIC_NO_BINDING"
    else:
        verdict = "NO_USABLE_TRANSFER"

    result = {
        "seed": SEED, "names": names, "target_id_tokens": target_k,
        "train_docs": TRAIN_DOCS, "dev_docs": DEV_DOCS, "test_docs": TEST_DOCS,
        "sender_layer_selection": list(SENDER_LAYER_SELECTION), "rank": RANK, "lr": LR,
        "trainable_parameter_count": adapter.param_count(),
        "epochs_run": len(train_loss_history), "stopped_early": stopped_early,
        "epochs_capped": epochs_capped, "wall_seconds": wall_s,
        "best_dev_loss_at_selection": best_dev,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_loss_history,
        "dev_acc_history": dev_acc_history,
        "test": {
            "correct": summarize(correct_nll, correct_exact, correct_top1),
            "shuffled_document": summarize(shuf_nll, shuf_exact, shuf_top1),
            "wrong_binding": summarize(wrong_nll, wrong_exact, wrong_top1),
            "zero_sender": summarize(zero_nll, zero_exact, zero_top1),
            "query_name_swap_secondary": summarize(swap_nll, swap_exact, swap_top1),
            "correct_minus_shuffled": shuffled_ci,
            "correct_minus_wrong_binding": wrong_binding_ci,
            "correct_minus_zero": zero_ci,
        },
        "finite": finite,
        "correct_beats_shuffled": correct_beats_shuffled,
        "correct_beats_wrong_binding": correct_beats_wrong_binding,
        "wrong_binding_rejected": correct_beats_wrong_binding,
        "test_closed_until_frozen": True,
        "verdict": verdict,
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")

    print(f"correct:          nll={correct_nll.mean().item():.4f} exact_acc={correct_exact.mean().item():.4f}")
    print(f"shuffled_document:nll={shuf_nll.mean().item():.4f} exact_acc={shuf_exact.mean().item():.4f}")
    print(f"wrong_binding:    nll={wrong_nll.mean().item():.4f} exact_acc={wrong_exact.mean().item():.4f}")
    print(f"zero_sender:      nll={zero_nll.mean().item():.4f} exact_acc={zero_exact.mean().item():.4f}")
    print(f"query_swap(sec):  nll={swap_nll.mean().item():.4f} exact_acc={swap_exact.mean().item():.4f}")
    print(f"correct-shuffled ci={shuffled_ci['bootstrap_95pct_ci']} beats={correct_beats_shuffled}")
    print(f"correct-wrong_binding ci={wrong_binding_ci['bootstrap_95pct_ci']} beats={correct_beats_wrong_binding}")
    print(f"correct-zero ci={zero_ci['bootstrap_95pct_ci']} beats={correct_beats_zero}")
    print(f"verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
