#!/usr/bin/env python3
"""Mini-GunPoint: 32-token global-pattern classification via KV-cache
handoff -- the opposite test from multi-key addressing, not another
bandwidth-exact-recovery rung.

Sender sees a 32-token binary sequence ("Sequence: A B B A A B ..."),
receiver sees NO sequence at all, only the transferred KV-cache plus a
fixed, sequence-independent query ("Class:"), and must classify the
GLOBAL temporal pattern: class 0 = first 16 tokens A-majority, second 16
B-majority; class 1 = the reverse. Both classes always contain exactly 16
A's and 16 B's total -- histogram/count alone carries zero class
information by construction; only positional/temporal structure does.
This directly tests "can KV handoff carry a global, order-dependent
property of a whole sequence" as opposed to "can it retrieve one specific
stored fact by key" (already shown to fail after three different adapter
architectures) or "can it losslessly recall every token" (bandwidth,
already shown to work near-perfectly at 1-2 tokens).

Same low-rank shared adapter as the original (successful) bandwidth
rungs -- not the per-position or query-conditioned addressing variants --
since this task has no key to address, matching the hypothesis that a
shared transform suffices for globally-consumed information.

Four standard controls (correct, shuffled_document, zero_sender,
no_prefix) plus the control specifically requested for this task:
position_shuffle -- the SAME document's 32 tokens (same multiset, same
16/16 split) randomly permuted before the SENDER encodes them, so the
class-defining half-wise structure is destroyed while the raw content
(which tokens appear) is unchanged. If the receiver's accuracy drops
toward chance under position_shuffle while staying high on correct, that
demonstrates genuine order-dependence, not just "detects presence of
extra A tokens somewhere". Primary gate: correct classification accuracy
significantly above chance (50%), correct beats shuffled_document (real
document content matters), and correct beats position_shuffle (temporal
order specifically matters, not just token identity/count).
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
SEQ_LEN = 32
HALF = SEQ_LEN // 2
K_RANGE = (10, 11, 12, 13, 14)  # first-half majority margin (k A's out of 16)
SENDER_LAYER_SELECTION = (1, 3, 5, 7, 9, 11)
RANK = 8
A_INIT_STD = 0.02
MAX_EPOCHS = 800  # convergence-completion: first pass was still_improving_at_cutoff
DEV_EVAL_EVERY = 5
STILL_IMPROVING_REL_THRESHOLD = 0.01
LR = 1e-2
CLASS_TOKENS = (" 0", " 1")  # verified single-BPE-token labels


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bucket(seq_str: str) -> str:
    value = int(sha(seq_str.encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def make_sequence(rng: random.Random, cls: int) -> list[str]:
    """cls=0: first half A-majority, second half B-majority.
    cls=1: reverse. Always exactly 16 A's and 16 B's total."""
    k = rng.choice(K_RANGE)
    if cls == 0:
        first_a, first_b = k, HALF - k
        second_a, second_b = HALF - k, k
    else:
        first_a, first_b = HALF - k, k
        second_a, second_b = k, HALF - k
    first = ["A"] * first_a + ["B"] * first_b
    second = ["A"] * second_a + ["B"] * second_b
    rng.shuffle(first)
    rng.shuffle(second)
    seq = first + second
    assert seq.count("A") == HALF and seq.count("B") == HALF
    return seq


def generate_pool(target_per_class: int) -> list[tuple[list[str], int]]:
    rng = random.Random(SEED)
    pool = []
    seen: set[str] = set()
    for cls in (0, 1):
        made = 0
        attempts = 0
        while made < target_per_class and attempts < target_per_class * 50:
            attempts += 1
            seq = make_sequence(rng, cls)
            key = "".join(seq)
            if key in seen:
                continue
            seen.add(key)
            pool.append((seq, cls))
            made += 1
        if made < target_per_class:
            raise RuntimeError(f"could not generate {target_per_class} unique class-{cls} sequences")
    return pool


def select_docs(pool: list[tuple[list[str], int]], split: str, count: int) -> list[tuple[list[str], int]]:
    chosen = sorted(
        (d for d in pool if bucket("".join(d[0])) == split),
        key=lambda d: sha("".join(d[0]).encode()),
    )
    if len(chosen) < count:
        raise RuntimeError(f"only {len(chosen)} eligible {split} docs")
    # keep class-balanced within the selected count, INTERLEAVED (not
    # concatenated) -- concatenating c0+c1 made roll-by-1 shuffling mostly
    # pair same-class neighbors (only 2 of 32 rows crossed the class
    # boundary), silently invalidating the shuffled_document control
    # (it scored ~96.875%, identical to correct, because it was usually
    # still the right class). Interleaving guarantees every roll-by-1 pair
    # crosses classes.
    c0 = [d for d in chosen if d[1] == 0][: count // 2]
    c1 = [d for d in chosen if d[1] == 1][: count // 2]
    if len(c0) < count // 2 or len(c1) < count // 2:
        raise RuntimeError(f"not enough class balance in {split}: {len(c0)} / {len(c1)}")
    interleaved = [d for pair in zip(c0, c1) for d in pair]
    assert [d[1] for d in interleaved] == [i % 2 for i in range(len(interleaved))]
    return interleaved


def seq_to_ids(tokenizer, seq: list[str]) -> list[int]:
    text = "Sequence:" + "".join(f" {tok}" for tok in seq)
    return tokenizer.encode(text, add_special_tokens=False)


def build_batch(sender, tokenizer, docs: list[tuple[list[str], int]]):
    tokenized = [seq_to_ids(tokenizer, seq) for seq, _ in docs]
    lens = {len(t) for t in tokenized}
    assert len(lens) == 1, f"prefix length not constant: {lens}"
    ids = torch.tensor(tokenized, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = sender(ids, use_cache=True)
    pkv = out.past_key_values
    keys = [pkv.layers[i].keys.clone().detach() for i in SENDER_LAYER_SELECTION]
    values = [pkv.layers[i].values.clone().detach() for i in SENDER_LAYER_SELECTION]
    target_ids = [tokenizer.encode(CLASS_TOKENS[cls], add_special_tokens=False)[0] for _, cls in docs]
    target = torch.tensor(target_ids, dtype=torch.long, device=DEVICE)
    return keys, values, target


def build_position_shuffle_batch(sender, tokenizer, docs: list[tuple[list[str], int]], seed: int):
    rng = random.Random(seed)
    shuffled_docs = []
    for seq, cls in docs:
        s = list(seq)
        rng.shuffle(s)
        shuffled_docs.append((s, cls))
    return build_batch(sender, tokenizer, shuffled_docs)


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

    def forward_keys(self, keys: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self._lowrank_correct(k, self.A_k[i], self.B_k[i]) for i, k in enumerate(keys)]

    def forward_values(self, values: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self._lowrank_correct(v, self.A_v[i], self.B_v[i]) for i, v in enumerate(values)]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_cache(keys, values) -> DynamicCache:
    cache = DynamicCache()
    for k, v in zip(keys, values):
        cache.update(k, v, layer_idx=len(cache.layers))
    return cache


def zero_like(keys: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.zeros_like(k) for k in keys]


def roll_batch(keys: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.roll(k, shifts=1, dims=0) for k in keys]


def query_ids(tokenizer, batch: int) -> torch.Tensor:
    ids = tokenizer.encode("Class:", return_tensors="pt").to(DEVICE)
    return ids.expand(batch, -1)


def next_token_logits(receiver, q_ids: torch.Tensor, cache: DynamicCache) -> torch.Tensor:
    out = receiver(q_ids, past_key_values=cache, use_cache=False)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bridge-out", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval().to(DEVICE)
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval().to(DEVICE)
    for m in (sender, receiver):
        for p in m.parameters():
            p.requires_grad_(False)

    pool = generate_pool(target_per_class=200)
    train_docs = select_docs(pool, "train", TRAIN_DOCS)
    dev_docs = select_docs(pool, "dev", DEV_DOCS)
    train_keys_set = {"".join(s) for s, _ in train_docs}
    dev_keys_set = {"".join(s) for s, _ in dev_docs}
    assert train_keys_set.isdisjoint(dev_keys_set)

    train_k, train_v, train_target = build_batch(sender, tokenizer, train_docs)
    dev_k, dev_v, dev_target = build_batch(sender, tokenizer, dev_docs)
    n_train = len(train_docs)
    n_dev = len(dev_docs)

    n_head = receiver.config.n_head
    head_dim = receiver.config.n_embd // n_head
    n_layers = len(SENDER_LAYER_SELECTION)

    adapter = LowRankKVAdapter(n_layers, head_dim, n_head, RANK, seed=SEED).to(DEVICE)
    opt = torch.optim.Adam(adapter.parameters(), lr=LR)
    print(f"seq_len={SEQ_LEN} trainable_params={adapter.param_count()}")

    tq = query_ids(tokenizer, n_train)
    k0 = adapter.forward_keys(train_k)
    v0 = adapter.forward_values(train_v)
    logits0 = next_token_logits(receiver, tq, make_cache(k0, v0))
    loss0 = torch.nn.functional.cross_entropy(logits0, train_target)
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
    dev_acc_history: list[float] = []
    dev_eval_epochs: list[int] = []
    train_loss_history: list[float] = []
    dq = query_ids(tokenizer, n_dev)

    def dev_metrics() -> tuple[float, float]:
        adapter.eval()
        with torch.no_grad():
            k = adapter.forward_keys(dev_k)
            v = adapter.forward_values(dev_v)
            logits = next_token_logits(receiver, dq, make_cache(k, v))
            loss = torch.nn.functional.cross_entropy(logits, dev_target).item()
            acc = (logits.argmax(-1) == dev_target).float().mean().item()
        adapter.train()
        return loss, acc

    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    for epoch in range(MAX_EPOCHS):
        adapter.train()
        opt.zero_grad()
        k = adapter.forward_keys(train_k)
        v = adapter.forward_values(train_v)
        logits = next_token_logits(receiver, tq, make_cache(k, v))
        loss = torch.nn.functional.cross_entropy(logits, train_target)
        loss.backward()
        opt.step()
        train_loss_history.append(loss.item())

        is_last = epoch == MAX_EPOCHS - 1
        if (epoch + 1) % DEV_EVAL_EVERY == 0 or is_last:
            dl, dacc = dev_metrics()
            dev_history.append(dl)
            dev_acc_history.append(dacc)
            dev_eval_epochs.append(epoch + 1)
            if dl < best_dev:
                best_dev = dl
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

        progress_path.write_text(json.dumps({
            "epochs_run": epoch + 1, "train_loss_history": train_loss_history,
            "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
            "dev_acc_history": dev_acc_history,
            "best_dev_loss_so_far": best_dev, "wall_seconds_so_far": time.time() - t0,
            "status": "training",
        }, indent=2) + "\n")

    adapter.load_state_dict(best_state)
    adapter.eval()
    wall_s = time.time() - t0

    if args.bridge_out is not None:
        torch.save({
            "state_dict": adapter.state_dict(),
            "sender_layer_selection": list(SENDER_LAYER_SELECTION),
            "rank": RANK, "n_head": n_head, "head_dim": head_dim,
            "train_docs_keys": sorted(train_keys_set), "dev_docs_keys": sorted(dev_keys_set),
        }, args.bridge_out)

    window = train_loss_history[-(DEV_EVAL_EVERY + 1):]
    still_improving = False
    if len(window) >= 2 and window[0] != 0:
        rel_decrease = (window[0] - window[-1]) / abs(window[0])
        still_improving = rel_decrease > STILL_IMPROVING_REL_THRESHOLD

    with torch.no_grad():
        correct_logits = next_token_logits(receiver, dq, make_cache(adapter.forward_keys(dev_k), adapter.forward_values(dev_v)))
        correct_nll, correct_top1 = nll_top1(correct_logits, dev_target)

        no_prefix_logits = next_token_logits(receiver, dq, DynamicCache())
        no_prefix_nll, no_prefix_top1 = nll_top1(no_prefix_logits, dev_target)

        zk, zv = zero_like(dev_k), zero_like(dev_v)
        zero_logits = next_token_logits(receiver, dq, make_cache(adapter.forward_keys(zk), adapter.forward_values(zv)))
        zero_nll, zero_top1 = nll_top1(zero_logits, dev_target)

        sk, sv = roll_batch(dev_k), roll_batch(dev_v)
        shuf_logits = next_token_logits(receiver, dq, make_cache(adapter.forward_keys(sk), adapter.forward_values(sv)))
        shuf_nll, shuf_top1 = nll_top1(shuf_logits, dev_target)

        ps_k, ps_v, ps_target = build_position_shuffle_batch(sender, tokenizer, dev_docs, seed=SEED + 999)
        assert torch.equal(ps_target, dev_target)  # same underlying labels, order destroyed not content
        ps_logits = next_token_logits(receiver, dq, make_cache(adapter.forward_keys(ps_k), adapter.forward_values(ps_v)))
        ps_nll, ps_top1 = nll_top1(ps_logits, dev_target)

    finite = all(torch.isfinite(x).all().item() for x in (correct_nll, no_prefix_nll, zero_nll, shuf_nll, ps_nll))

    def advantage(other_nll):
        return (other_nll - correct_nll).tolist()

    shuffled_advantage = advantage(shuf_nll)
    zero_advantage = advantage(zero_nll)
    no_prefix_advantage = advantage(no_prefix_nll)
    ps_advantage = advantage(ps_nll)

    shuffled_ci = bootstrap_ci(shuffled_advantage, SEED)
    zero_ci = bootstrap_ci(zero_advantage, SEED + 1)
    no_prefix_ci = bootstrap_ci(no_prefix_advantage, SEED + 2)
    ps_ci = bootstrap_ci(ps_advantage, SEED + 3)

    correct_acc = correct_top1.mean().item()
    chance_baseline = 0.5
    accuracy_above_chance = correct_acc > chance_baseline
    beats_shuffled_doc = finite and shuffled_ci[0] > 0
    beats_position_shuffle = finite and ps_ci[0] > 0
    signal = accuracy_above_chance and beats_shuffled_doc and beats_position_shuffle

    def summarize(nll, top1):
        return {"nll": nll.mean().item(), "accuracy": top1.mean().item()}

    result = {
        "seed": SEED, "seq_len": SEQ_LEN, "k_range": list(K_RANGE),
        "sender_layer_selection": list(SENDER_LAYER_SELECTION), "rank": RANK,
        "trainable_parameter_count": adapter.param_count(),
        "epochs_run": len(train_loss_history), "wall_seconds": wall_s,
        "best_dev_loss_at_selection": best_dev,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
        "dev_acc_history": dev_acc_history,
        "still_improving_at_cutoff": still_improving,
        "mechanism": "low_rank_residual_kv_adapter_batched_minigunpoint_32token",
        "dev": {
            "correct": summarize(correct_nll, correct_top1),
            "shuffled_document": summarize(shuf_nll, shuf_top1),
            "position_shuffle": summarize(ps_nll, ps_top1),
            "zero_sender": summarize(zero_nll, zero_top1),
            "no_prefix": summarize(no_prefix_nll, no_prefix_top1),
            "shuffled_minus_correct_nll": sum(shuffled_advantage) / len(shuffled_advantage),
            "shuffled_minus_correct_bootstrap_95pct_ci": shuffled_ci,
            "position_shuffle_minus_correct_nll": sum(ps_advantage) / len(ps_advantage),
            "position_shuffle_minus_correct_bootstrap_95pct_ci": ps_ci,
            "zero_sender_minus_correct_nll": sum(zero_advantage) / len(zero_advantage),
            "zero_sender_minus_correct_bootstrap_95pct_ci": zero_ci,
            "no_prefix_minus_correct_nll": sum(no_prefix_advantage) / len(no_prefix_advantage),
            "no_prefix_minus_correct_bootstrap_95pct_ci": no_prefix_ci,
        },
        "finite": finite,
        "chance_baseline": chance_baseline,
        "accuracy_above_chance": accuracy_above_chance,
        "beats_shuffled_document": beats_shuffled_doc,
        "beats_position_shuffle": beats_position_shuffle,
        "verdict": (
            "MINIGUNPOINT_TRAINING_INCONCLUSIVE" if still_improving else
            "MINIGUNPOINT_GLOBAL_PATTERN_TRANSFER_SUPPORTED" if signal else
            "MINIGUNPOINT_GLOBAL_PATTERN_TRANSFER_NOT_SUPPORTED"
        ),
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(f"correct:          nll={result['dev']['correct']['nll']:.4f} acc={result['dev']['correct']['accuracy']:.4f}")
    print(f"shuffled_document:nll={result['dev']['shuffled_document']['nll']:.4f} acc={result['dev']['shuffled_document']['accuracy']:.4f}")
    print(f"position_shuffle: nll={result['dev']['position_shuffle']['nll']:.4f} acc={result['dev']['position_shuffle']['accuracy']:.4f}")
    print(f"zero_sender:      nll={result['dev']['zero_sender']['nll']:.4f} acc={result['dev']['zero_sender']['accuracy']:.4f}")
    print(f"no_prefix:        nll={result['dev']['no_prefix']['nll']:.4f} acc={result['dev']['no_prefix']['accuracy']:.4f}")
    print(f"shuffled_minus_correct={result['dev']['shuffled_minus_correct_nll']:.4f} ci={shuffled_ci}")
    print(f"position_shuffle_minus_correct={result['dev']['position_shuffle_minus_correct_nll']:.4f} ci={ps_ci}")
    print(f"verdict={result['verdict']} wall_s={wall_s:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
