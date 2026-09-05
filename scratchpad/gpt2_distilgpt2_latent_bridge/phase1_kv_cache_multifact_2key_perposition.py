#!/usr/bin/env python3
"""Multi-fact KV-cache addressing, K=2 keys -- not just "more tokens", but
"can the receiver select the queried value out of several stored in the
same cache".

Prior steps showed the KV-cache mechanism carries a single fact (1 token:
100% top-1; 2 tokens: 96.9% exact) with almost no bandwidth loss. This
tests a qualitatively different question: sender sees TWO (key, value)
pairs in one context; receiver is queried about only ONE key and must
retrieve the matching value from the SAME transferred cache that also
contains the other pair. Names are fixed placeholders ("Person1",
"Person2"), not drawn from a pool, so the receiver-visible query text is
constant across every document except for which slot is asked (same
discipline as every prior rung: only the injected content should carry
information, not the visible text). Only the two codes vary per document,
each drawn from the verified single-BPE-token 3-digit pool (100-999, 405
eligible values), constrained to be distinct within a document (otherwise
"wrong key" would trivially pass).

Prefix (sender-only): "Person1's code is {code1}. Person2's code is {code2}."
Query (receiver-visible, one of two fixed strings): "Person1's code is" or
"Person2's code is". Four evaluation conditions per document, all sharing
one transferred cache:
  - correct:      query_i -> target_i (i in {1,2}), the real addressing task
  - wrong_key:    query_i -> target_j (j != i) -- does asking about the
                  OTHER key still retrieve target_i (bad: means the
                  receiver ignores the query and just regurgitates "a"
                  stored value) or correctly fail to match target_i (good:
                  means retrieval is actually keyed by the query)? This is
                  the critical control the single-fact rungs could not
                  test at all, since there was only ever one thing to ask.
  - shuffled_doc: a different document's cache entirely, same query_i,
                  scored against this document's target_i (same convention
                  as every prior shuffled-document control)
  - zero_sender / no_prefix: same as every prior rung

Primary gate: correct (diagonal) beats wrong_key (off-diagonal) with CI
excluding zero, AND correct beats shuffled_document with CI excluding
zero. Passing shuffled_document alone is not sufficient here -- a receiver
that just memorizes "the codes seen in this document, in no particular
order" could beat shuffled_document without being able to address by key
at all; only beating wrong_key demonstrates addressing.

The 2-key shared-adapter run (`phase1_kv_cache_multifact_2key_batched.py`)
completely failed to address: `wrong_key_minus_correct` CI
`[-0.316, 0.312]` (cannot tell which key was asked at all), despite a real
`zero_sender`/`no_prefix` gap showing the cache does carry usable content
overall. Refined hypothesis (attention retrieves via Q<->K match; K is
"where a fact lives", V is "what's stored there"): the failure implicates
K specifically, not V, since content is present but not selectable.
Asymmetric fix, not a uniform one: K gets a separate (A,B) pair PER FIXED
PREFIX POSITION (not per segment inferred at runtime -- the template is
fixed-length and every position's role is constant across documents by
construction), while V keeps the original single shared (A,B) pair per
layer, unchanged from the failed run. This multiplies K's parameter count
by the prefix length (~14-16x) while leaving V's untouched -- roughly half
the parameter growth of making both per-position, and a more direct test
of "the failure is in K, not V" than changing both at once. Total on the
same 64-document
/ 128-sub-example training budget -- a real overfitting risk this script
does not resolve in advance, only tests directly, consistent with this
project's discipline of running the experiment rather than assuming its
outcome.
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
TRAIN_DOCS = 64
DEV_DOCS = 32
CODE_MIN, CODE_MAX = 100, 999
SENDER_LAYER_SELECTION = (1, 3, 5, 7, 9, 11)
RANK = 8
A_INIT_STD = 0.02
MAX_EPOCHS = 300
DEV_EVAL_EVERY = 5
STILL_IMPROVING_REL_THRESHOLD = 0.01
LR = 1e-2
NAMES = ("Person1", "Person2")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bucket(doc_key: str) -> str:
    value = int(sha(doc_key.encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def single_token_codes(tokenizer) -> list[int]:
    return [c for c in range(CODE_MIN, CODE_MAX + 1) if len(tokenizer.encode(f" {c}", add_special_tokens=False)) == 1]


def select_pairs(tokenizer, split: str, count: int) -> list[tuple[int, int]]:
    """Deterministic doc = (code1, code2), code1 != code2, both single-token.
    Bucketed on the pair's own hash, disjoint across splits by construction."""
    eligible = single_token_codes(tokenizer)
    rng = random.Random(SEED)
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []
    pool = list(eligible)
    while len(candidates) < 20000 and len(seen) < len(pool) * (len(pool) - 1):
        c1 = rng.choice(pool)
        c2 = rng.choice(pool)
        if c1 == c2 or (c1, c2) in seen:
            continue
        seen.add((c1, c2))
        candidates.append((c1, c2))
    chosen = sorted(
        (p for p in candidates if bucket(f"{p[0]}|{p[1]}") == split),
        key=lambda p: sha(f"{p[0]}|{p[1]}".encode()),
    )
    if len(chosen) < count:
        raise RuntimeError(f"only {len(chosen)} eligible {split} pairs")
    return chosen[:count]


def build_batch(sender, tokenizer, pairs: list[tuple[int, int]]):
    """Returns per-layer batched keys/values (batch=n_docs) for the shared
    cache, plus target ids (batch, 2) -- target[:,0] is code1 (Person1's
    value), target[:,1] is code2 (Person2's value).

    Single batched sender forward pass, not a per-document loop: every
    prefix has the same tokenized length by construction (fixed template,
    single-BPE-token codes), so all N documents stack cleanly along the
    batch dimension for one call. Flagged directly after the 2-key run's
    ~90s per-document data-prep phase turned out to dominate wall time
    despite the training loop itself being batched -- fixed here rather
    than left for the next script to repeat."""
    texts = [f"{NAMES[0]}'s code is {c1}. {NAMES[1]}'s code is {c2}." for c1, c2 in pairs]
    tokenized = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    lens = {len(t) for t in tokenized}
    assert len(lens) == 1, f"prefix length not constant: {lens}"
    ids = torch.tensor(tokenized, dtype=torch.long, device=DEVICE)  # (batch, seq)
    with torch.no_grad():
        out = sender(ids, use_cache=True)
    pkv = out.past_key_values
    keys = [pkv.layers[i].keys.clone().detach() for i in SENDER_LAYER_SELECTION]
    values = [pkv.layers[i].values.clone().detach() for i in SENDER_LAYER_SELECTION]

    targets = []
    for code1, code2 in pairs:
        t1 = tokenizer.encode(f" {code1}", add_special_tokens=False)
        t2 = tokenizer.encode(f" {code2}", add_special_tokens=False)
        assert len(t1) == 1 and len(t2) == 1
        targets.append([t1[0], t2[0]])
    target = torch.tensor(targets, dtype=torch.long, device=DEVICE)  # (batch, 2)
    return keys, values, target


class PerPositionLowRankKVAdapter(nn.Module):
    """Asymmetric fix, not a uniform one: addressing in real attention is
    a Q<->K match (K is "where a fact lives", V is "what's stored there");
    the 2-key failure showed content is present (zero_sender/no_prefix gap
    is real) but not selectable by key, which implicates K specifically,
    not V. So K gets a separate (A,B) pair PER PREFIX POSITION (position p's
    transform can differ from position q's, giving the receiver's
    query-dependent attention something position-specific to discriminate),
    while V keeps the ORIGINAL single shared (A,B) pair per layer (V only
    needs cross-model translation, not per-entity addressability, per this
    hypothesis) -- half the parameter growth of making both per-position,
    and a more direct test of "the failure is in K, not V". B is
    zero-initialized everywhere (init is still the identity/raw baseline)."""

    def __init__(self, n_layers: int, prefix_len: int, head_dim: int, n_head: int, rank: int, seed: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.prefix_len = prefix_len
        width = n_head * head_dim
        g = torch.Generator().manual_seed(seed)
        self.A_k = nn.ParameterList([nn.Parameter(torch.randn(prefix_len, width, rank, generator=g) * A_INIT_STD) for _ in range(n_layers)])
        self.B_k = nn.ParameterList([nn.Parameter(torch.zeros(prefix_len, rank, width)) for _ in range(n_layers)])
        self.A_v = nn.ParameterList([nn.Parameter(torch.randn(width, rank, generator=g) * A_INIT_STD) for _ in range(n_layers)])
        self.B_v = nn.ParameterList([nn.Parameter(torch.zeros(rank, width)) for _ in range(n_layers)])

    def _lowrank_correct_perposition(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_head, seq, head_dim); A: (seq, width, rank); B: (seq, rank, width)
        b, h, s, d = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(b, s, h * d)  # (batch, seq, width)
        corr = torch.einsum("bpw,pwr->bpr", flat, A)
        corr = torch.einsum("bpr,prw->bpw", corr, B)
        out_flat = flat + corr
        return out_flat.reshape(b, s, h, d).permute(0, 2, 1, 3)

    def _lowrank_correct_shared(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_head, seq, head_dim); A: (width, rank); B: (rank, width) -- same transform at every position
        b, h, s, d = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(b * s, h * d)
        correction = (flat @ A) @ B
        out_flat = flat + correction
        return out_flat.reshape(b, s, h, d).permute(0, 2, 1, 3)

    def forward_keys(self, keys: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self._lowrank_correct_perposition(k, self.A_k[i], self.B_k[i]) for i, k in enumerate(keys)]

    def forward_values(self, values: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self._lowrank_correct_shared(v, self.A_v[i], self.B_v[i]) for i, v in enumerate(values)]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_cache(keys, values) -> DynamicCache:
    cache = DynamicCache()
    for k, v in zip(keys, values):
        cache.update(k, v, layer_idx=len(cache.layers))
    return cache


def repeat_kv(keys: list[torch.Tensor], times: int) -> list[torch.Tensor]:
    return [k.repeat_interleave(times, dim=0) for k in keys]


def zero_like(keys: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.zeros_like(k) for k in keys]


def roll_batch(keys: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.roll(k, shifts=1, dims=0) for k in keys]


def query_ids_for(tokenizer, slot: int, batch: int) -> torch.Tensor:
    ids = tokenizer.encode(f"{NAMES[slot]}'s code is", return_tensors="pt").to(DEVICE)
    return ids.expand(batch, -1)


def next_token_logits(receiver, query_ids: torch.Tensor, cache: DynamicCache) -> torch.Tensor:
    out = receiver(query_ids, past_key_values=cache, use_cache=False)
    return out.logits[:, -1, :].float()


def nll_top1(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    top1 = (logits.argmax(-1) == target).float()
    return nll, top1


def bootstrap_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return [means[249], means[9749]]


def build_training_batch(keys, values, target, tokenizer, n_docs: int):
    """Two sub-examples per document (query slot 0 and slot 1), same cache
    repeated for both, so the adapter must learn one shared cache that
    answers both queries correctly."""
    k2 = repeat_kv(keys, 2)
    v2 = repeat_kv(values, 2)
    q0 = query_ids_for(tokenizer, 0, n_docs)
    q1 = query_ids_for(tokenizer, 1, n_docs)
    query = torch.cat([q0, q1], dim=0)  # (2*n_docs, seq)
    tgt = torch.cat([target[:, 0], target[:, 1]], dim=0)  # (2*n_docs,)
    return k2, v2, query, tgt


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

    train_pairs = select_pairs(tokenizer, "train", TRAIN_DOCS)
    dev_pairs = select_pairs(tokenizer, "dev", DEV_DOCS)
    assert set(train_pairs).isdisjoint(dev_pairs)

    train_k, train_v, train_target = build_batch(sender, tokenizer, train_pairs)
    dev_k, dev_v, dev_target = build_batch(sender, tokenizer, dev_pairs)
    n_train = len(train_pairs)
    n_dev = len(dev_pairs)

    n_head = receiver.config.n_head
    head_dim = receiver.config.n_embd // n_head
    n_layers = len(SENDER_LAYER_SELECTION)
    prefix_len = train_k[0].shape[2]

    adapter = PerPositionLowRankKVAdapter(n_layers, prefix_len, head_dim, n_head, RANK, seed=SEED).to(DEVICE)
    opt = torch.optim.Adam(adapter.parameters(), lr=LR)
    print(f"prefix_len={prefix_len} trainable_params={adapter.param_count()}")

    tk2, tv2, tquery, ttgt = build_training_batch(train_k, train_v, train_target, tokenizer, n_train)
    k0 = adapter.forward_keys(tk2)
    v0 = adapter.forward_values(tv2)
    logits0 = next_token_logits(receiver, tquery, make_cache(k0, v0))
    loss0 = torch.nn.functional.cross_entropy(logits0, ttgt)
    loss0.backward()
    b_grad_nonzero = any(adapter.B_k[i].grad.abs().max().item() > 0 for i in range(n_layers)) or \
                      any(adapter.B_v[i].grad.abs().max().item() > 0 for i in range(n_layers))
    if not b_grad_nonzero:
        raise RuntimeError("KV low-rank adapter B has no learning signal at init")
    opt.zero_grad()

    t0 = time.time()
    best_dev = math.inf
    best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
    dev_history: list[float] = []
    dev_eval_epochs: list[int] = []
    train_loss_history: list[float] = []

    def eval_arm(keys, values, slot: int, target_col: int) -> tuple[list[float], list[float]]:
        """Query slot `slot`, score against target[:, target_col]."""
        q = query_ids_for(tokenizer, slot, keys[0].shape[0])
        with torch.no_grad():
            k = adapter.forward_keys(keys)
            v = adapter.forward_values(values)
            logits = next_token_logits(receiver, q, make_cache(k, v))
        nll, top1 = nll_top1(logits, dev_target[:, target_col])
        return nll.tolist(), top1.tolist()

    def dev_loss() -> float:
        adapter.eval()
        with torch.no_grad():
            k2, v2, query, tgt = build_training_batch(dev_k, dev_v, dev_target, tokenizer, n_dev)
            k = adapter.forward_keys(k2)
            v = adapter.forward_values(v2)
            logits = next_token_logits(receiver, query, make_cache(k, v))
            loss = torch.nn.functional.cross_entropy(logits, tgt).item()
        adapter.train()
        return loss

    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    for epoch in range(MAX_EPOCHS):
        adapter.train()
        opt.zero_grad()
        tk2, tv2, tquery, ttgt = build_training_batch(train_k, train_v, train_target, tokenizer, n_train)
        k = adapter.forward_keys(tk2)
        v = adapter.forward_values(tv2)
        logits = next_token_logits(receiver, tquery, make_cache(k, v))
        loss = torch.nn.functional.cross_entropy(logits, ttgt)
        loss.backward()
        opt.step()
        train_loss_history.append(loss.item())

        is_last = epoch == MAX_EPOCHS - 1
        if (epoch + 1) % DEV_EVAL_EVERY == 0 or is_last:
            dl = dev_loss()
            dev_history.append(dl)
            dev_eval_epochs.append(epoch + 1)
            if dl < best_dev:
                best_dev = dl
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

        progress_path.write_text(json.dumps({
            "epochs_run": epoch + 1, "train_loss_history": train_loss_history,
            "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
            "best_dev_loss_so_far": best_dev, "wall_seconds_so_far": time.time() - t0,
            "status": "training",
        }, indent=2) + "\n")

    adapter.load_state_dict(best_state)
    adapter.eval()
    wall_s = time.time() - t0

    window = train_loss_history[-(DEV_EVAL_EVERY + 1):]
    still_improving = False
    if len(window) >= 2 and window[0] != 0:
        rel_decrease = (window[0] - window[-1]) / abs(window[0])
        still_improving = rel_decrease > STILL_IMPROVING_REL_THRESHOLD

    # Evaluate both query slots against both targets on the real dev cache
    # (correct = diagonal, wrong_key = off-diagonal), plus shuffled-doc and
    # baseline controls scored on slot 0's query/target for simplicity.
    correct_nll, correct_top1 = [], []
    for slot in (0, 1):
        nll, top1 = eval_arm(dev_k, dev_v, slot, slot)
        correct_nll.extend(nll)
        correct_top1.extend(top1)

    wrong_key_nll, wrong_key_top1 = [], []
    for slot, other_col in ((0, 1), (1, 0)):
        nll, top1 = eval_arm(dev_k, dev_v, slot, other_col)
        wrong_key_nll.extend(nll)
        wrong_key_top1.extend(top1)

    shuf_k = roll_batch(dev_k)
    shuf_v = roll_batch(dev_v)
    shuffled_nll, shuffled_top1 = [], []
    for slot in (0, 1):
        nll, top1 = eval_arm(shuf_k, shuf_v, slot, slot)
        shuffled_nll.extend(nll)
        shuffled_top1.extend(top1)

    zk = zero_like(dev_k)
    zv = zero_like(dev_v)
    zero_nll, zero_top1 = [], []
    for slot in (0, 1):
        nll, top1 = eval_arm(zk, zv, slot, slot)
        zero_nll.extend(nll)
        zero_top1.extend(top1)

    no_prefix_nll, no_prefix_top1 = [], []
    for slot in (0, 1):
        q = query_ids_for(tokenizer, slot, n_dev)
        with torch.no_grad():
            logits = next_token_logits(receiver, q, DynamicCache())
        nll, top1 = nll_top1(logits, dev_target[:, slot])
        no_prefix_nll.extend(nll.tolist())
        no_prefix_top1.extend(top1.tolist())

    finite = all(math.isfinite(x) for rows in (correct_nll, wrong_key_nll, shuffled_nll, zero_nll, no_prefix_nll) for x in rows)

    content_advantage = [wk - c for wk, c in zip(wrong_key_nll, correct_nll)]
    shuffled_advantage = [s - c for s, c in zip(shuffled_nll, correct_nll)]
    zero_advantage = [z - c for z, c in zip(zero_nll, correct_nll)]
    no_prefix_advantage = [n - c for n, c in zip(no_prefix_nll, correct_nll)]

    content_ci = bootstrap_ci(content_advantage, SEED)
    shuffled_ci = bootstrap_ci(shuffled_advantage, SEED + 1)
    zero_ci = bootstrap_ci(zero_advantage, SEED + 2)
    no_prefix_ci = bootstrap_ci(no_prefix_advantage, SEED + 3)

    addressing_signal = finite and content_ci[0] > 0
    content_specific_signal = finite and shuffled_ci[0] > 0
    signal = addressing_signal and content_specific_signal

    def summarize(nll_list, top1_list):
        return {"nll": sum(nll_list) / len(nll_list), "top1": sum(top1_list) / len(top1_list)}

    result = {
        "seed": SEED, "sender_layer_selection": list(SENDER_LAYER_SELECTION), "rank": RANK,
        "n_keys": 2, "names": list(NAMES),
        "train_pairs": train_pairs, "dev_pairs": dev_pairs,
        "trainable_parameter_count": adapter.param_count(),
        "epochs_run": len(train_loss_history), "wall_seconds": wall_s,
        "best_dev_loss_at_selection": best_dev,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
        "still_improving_at_cutoff": still_improving,
        "mechanism": "per_position_low_rank_residual_kv_adapter_batched_2key_addressing",
        "dev": {
            "correct": summarize(correct_nll, correct_top1),
            "wrong_key": summarize(wrong_key_nll, wrong_key_top1),
            "shuffled_document": summarize(shuffled_nll, shuffled_top1),
            "zero_sender": summarize(zero_nll, zero_top1),
            "no_prefix": summarize(no_prefix_nll, no_prefix_top1),
            "wrong_key_minus_correct_nll": sum(content_advantage) / len(content_advantage),
            "wrong_key_minus_correct_bootstrap_95pct_ci": content_ci,
            "shuffled_minus_correct_nll": sum(shuffled_advantage) / len(shuffled_advantage),
            "shuffled_minus_correct_bootstrap_95pct_ci": shuffled_ci,
            "zero_sender_minus_correct_nll": sum(zero_advantage) / len(zero_advantage),
            "zero_sender_minus_correct_bootstrap_95pct_ci": zero_ci,
            "no_prefix_minus_correct_nll": sum(no_prefix_advantage) / len(no_prefix_advantage),
            "no_prefix_minus_correct_bootstrap_95pct_ci": no_prefix_ci,
        },
        "finite": finite,
        "addressing_signal_wrong_key_beats": addressing_signal,
        "content_specific_signal_shuffled_beats": content_specific_signal,
        "verdict": (
            "KV_2KEY_PERPOSITION_TRAINING_INCONCLUSIVE" if still_improving else
            "KV_2KEY_PERPOSITION_ADDRESSING_SUPPORTED" if signal else
            "KV_2KEY_PERPOSITION_ADDRESSING_NOT_SUPPORTED"
        ),
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(f"correct:    nll={result['dev']['correct']['nll']:.4f} top1={result['dev']['correct']['top1']:.4f}")
    print(f"wrong_key:  nll={result['dev']['wrong_key']['nll']:.4f} top1={result['dev']['wrong_key']['top1']:.4f}")
    print(f"shuffled:   nll={result['dev']['shuffled_document']['nll']:.4f} top1={result['dev']['shuffled_document']['top1']:.4f}")
    print(f"zero:       nll={result['dev']['zero_sender']['nll']:.4f} top1={result['dev']['zero_sender']['top1']:.4f}")
    print(f"no_prefix:  nll={result['dev']['no_prefix']['nll']:.4f} top1={result['dev']['no_prefix']['top1']:.4f}")
    print(f"wrong_key_minus_correct={result['dev']['wrong_key_minus_correct_nll']:.4f} ci={content_ci}")
    print(f"shuffled_minus_correct={result['dev']['shuffled_minus_correct_nll']:.4f} ci={shuffled_ci}")
    print(f"verdict={result['verdict']} wall_s={wall_s:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
