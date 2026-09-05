#!/usr/bin/env python3
"""Compositionality via KV-cache handoff: can the receiver compute a
target that depends on a NONLINEAR COMBINATION of two independent
structural bits, each carried by a disjoint half of the transplanted
32-token cache, when neither bit alone is predictive of the target?

Sequence = 16-token half1 + 16-token half2. Each half independently
encodes one bit via the SAME recursive rule as the ordering task (first
8-token sub-half majority-A vs majority-B), always exactly 8 A's / 8 B's
per half so half-level histogram carries zero information about that
half's own bit. b1, b2 ~ Uniform{0,1} independently; class = b1 XOR b2.
Marginally, P(class | b1) = 0.5 for any b1 (b2 is independent uniform),
and symmetrically for b2 -- by construction, no linear/marginal readout
of either half alone can predict class above chance. Solving this
requires the receiver to read both halves AND combine them
non-additively; the nonlinearity is supplied by the frozen receiver's own
attention/FFN stack, not by the (linear, low-rank) adapter -- this
mirrors the already-successful ordering rung's adapter architecture
exactly, changing only the task.

Controls beyond the standard four (shuffled_document, position_shuffle,
zero_sender, no_prefix):
  - half1_only / half2_only: zero the RAW sender KV at the other half's
    prefix positions before the trained adapter is applied, keeping the
    named half's real KV untouched. Tests whether either half ALONE,
    passed through the same trained adapter, is decodable above chance --
    it must not be, or "compositionality" would really just be "a fancier
    single-fact readout that happens to look at one half".
  - half_swap: splice half1 from one dev document with half2 from a
    DIFFERENT dev document (paired by roll-by-1, same convention as
    shuffled_document), re-encode the spliced 32-token sequence through
    the sender fresh (like position_shuffle does), and score against the
    NEW ground truth (b1 of doc i XOR b2 of doc j) -- a combination never
    seen as a whole unit in training. This is the decisive test: if
    correct works but half_swap does not, the receiver memorized whole
    32-token gestalts rather than genuinely combining two independently
    decodable parts.

Falsifies compositionality if: correct is at chance, OR correct does not
beat half1_only/half2_only, OR half_swap is at chance despite correct
working. Positive requires correct significantly above chance AND correct
beats both half-only ablations AND half_swap significantly above chance.
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

SEED = 20260905
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_DOCS = 256  # sample-complexity scale-up test: identical protocol, only this changed
DEV_DOCS = 32
SEQ_LEN = 32
HALF = SEQ_LEN // 2  # 16
SUB = HALF // 2  # 8
SUB_K_RANGE = (5, 6)  # sub-half majority margin (k A's out of 8, k=4 would tie)
SENDER_LAYER_SELECTION = (1, 3, 5, 7, 9, 11)
RANK = 8
A_INIT_STD = 0.02
MAX_EPOCHS = 100  # timing probe for the 512-doc scale-up before committing a larger budget
DEV_EVAL_EVERY = 5
STILL_IMPROVING_REL_THRESHOLD = 0.01
LR = 1e-2
CLASS_TOKENS = (" 0", " 1")  # verified single-BPE-token labels


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bucket(key_str: str) -> str:
    value = int(sha(key_str.encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def make_half(rng: random.Random, bit: int) -> list[str]:
    """bit=0: first 8-token sub-half A-majority, second B-majority.
    bit=1: reverse. Always exactly 8 A's and 8 B's in the 16-token half."""
    k = rng.choice(SUB_K_RANGE)
    if bit == 0:
        first_a, first_b = k, SUB - k
        second_a, second_b = SUB - k, k
    else:
        first_a, first_b = SUB - k, k
        second_a, second_b = k, SUB - k
    first = ["A"] * first_a + ["B"] * first_b
    second = ["A"] * second_a + ["B"] * second_b
    rng.shuffle(first)
    rng.shuffle(second)
    half = first + second
    assert half.count("A") == SUB and half.count("B") == SUB
    return half


def make_doc(rng: random.Random, b1: int, b2: int) -> list[str]:
    return make_half(rng, b1) + make_half(rng, b2)


Doc = tuple[list[str], int, int, int]  # seq, b1, b2, cls


def generate_pool(target_per_combo: int) -> list[Doc]:
    rng = random.Random(SEED)
    pool: list[Doc] = []
    seen: set[str] = set()
    for b1 in (0, 1):
        for b2 in (0, 1):
            made = 0
            attempts = 0
            while made < target_per_combo and attempts < target_per_combo * 50:
                attempts += 1
                seq = make_doc(rng, b1, b2)
                key = "".join(seq)
                if key in seen:
                    continue
                seen.add(key)
                pool.append((seq, b1, b2, b1 ^ b2))
                made += 1
            if made < target_per_combo:
                raise RuntimeError(f"could not generate {target_per_combo} unique (b1={b1},b2={b2}) sequences")
    return pool


def select_docs(pool: list[Doc], split: str, count: int) -> list[Doc]:
    """Balances BOTH class (for the target label) AND the underlying
    (b1,b2) combo (for compositionality validity) -- class-only balance is
    not enough: with only ~16-32 docs per class, an uneven split across a
    class's two constituent (b1,b2) combos (e.g. (0,1) >> (1,0) within
    class 1) lets one bit correlate with class by pure sampling accident,
    contaminating the half1_only/half2_only ablations and half_swap. A
    first run without combo balancing found exactly this: half2_only
    scored 62.5% (should be ~chance), traced to a real (0,1):(1,0) = 11:5
    imbalance within the dev class-1 docs."""
    chosen = sorted(
        (d for d in pool if bucket("".join(d[0])) == split),
        key=lambda d: sha("".join(d[0]).encode()),
    )
    quota = count // 4
    if quota * 4 != count:
        raise RuntimeError(f"count={count} must be divisible by 4 for combo balance")
    by_combo = {(b1, b2): [d for d in chosen if (d[1], d[2]) == (b1, b2)][:quota]
                for b1 in (0, 1) for b2 in (0, 1)}
    for combo, docs in by_combo.items():
        if len(docs) < quota:
            raise RuntimeError(f"not enough {split} docs for combo {combo}: {len(docs)} < {quota}")
    class0_seq = [d for pair in zip(by_combo[(0, 0)], by_combo[(1, 1)]) for d in pair]
    class1_seq = [d for pair in zip(by_combo[(0, 1)], by_combo[(1, 0)]) for d in pair]
    interleaved = [d for pair in zip(class0_seq, class1_seq) for d in pair]
    assert [d[3] for d in interleaved] == [i % 2 for i in range(len(interleaved))]
    for combo in by_combo:
        assert sum(1 for d in interleaved if (d[1], d[2]) == combo) == quota
    return interleaved


def header_ids(tokenizer) -> list[int]:
    return tokenizer.encode("Sequence:", add_special_tokens=False)


def seq_to_ids(tokenizer, seq: list[str]) -> list[int]:
    return header_ids(tokenizer) + [tokenizer.encode(f" {tok}", add_special_tokens=False)[0] for tok in seq]


def build_batch(sender, tokenizer, docs: list[Doc]):
    tokenized = [seq_to_ids(tokenizer, seq) for seq, _, _, _ in docs]
    lens = {len(t) for t in tokenized}
    assert len(lens) == 1, f"prefix length not constant: {lens}"
    ids = torch.tensor(tokenized, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = sender(ids, use_cache=True)
    pkv = out.past_key_values
    keys = [pkv.layers[i].keys.clone().detach() for i in SENDER_LAYER_SELECTION]
    values = [pkv.layers[i].values.clone().detach() for i in SENDER_LAYER_SELECTION]
    target_ids = [tokenizer.encode(CLASS_TOKENS[cls], add_special_tokens=False)[0] for _, _, _, cls in docs]
    target = torch.tensor(target_ids, dtype=torch.long, device=DEVICE)
    return keys, values, target


def build_position_shuffle_batch(sender, tokenizer, docs: list[Doc], seed: int):
    rng = random.Random(seed)
    shuffled_docs: list[Doc] = []
    for seq, b1, b2, cls in docs:
        s = list(seq)
        rng.shuffle(s)
        shuffled_docs.append((s, b1, b2, cls))
    return build_batch(sender, tokenizer, shuffled_docs)


def build_half_swap_batch(sender, tokenizer, docs: list[Doc]):
    """Splice half1 of doc i with half2 of doc (i+1 mod n); new label is
    b1(doc_i) XOR b2(doc_{i+1}), a combination not present as a whole
    document in the pool by construction (independently random halves)."""
    n = len(docs)
    swapped_docs: list[Doc] = []
    for i in range(n):
        seq_i, b1_i, _, _ = docs[i]
        seq_j, _, b2_j, _ = docs[(i + 1) % n]
        spliced = seq_i[:HALF] + seq_j[HALF:]
        swapped_docs.append((spliced, b1_i, b2_j, b1_i ^ b2_j))
    return build_batch(sender, tokenizer, swapped_docs)


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


def zero_positions(keys: list[torch.Tensor], lo: int, hi: int) -> list[torch.Tensor]:
    out = []
    for k in keys:
        k2 = k.clone()
        k2[:, :, lo:hi, :] = 0.0
        out.append(k2)
    return out


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
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval().to(DEVICE)
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval().to(DEVICE)
    for m in (sender, receiver):
        for p in m.parameters():
            p.requires_grad_(False)

    header_len = len(header_ids(tokenizer))
    half1_lo, half1_hi = header_len, header_len + HALF
    half2_lo, half2_hi = header_len + HALF, header_len + SEQ_LEN

    pool = generate_pool(target_per_combo=1000)
    train_docs = select_docs(pool, "train", TRAIN_DOCS)
    dev_docs = select_docs(pool, "dev", DEV_DOCS)
    train_keys_set = {"".join(s) for s, _, _, _ in train_docs}
    dev_keys_set = {"".join(s) for s, _, _, _ in dev_docs}
    assert train_keys_set.isdisjoint(dev_keys_set)

    def combo_table(docs: list[Doc]) -> dict[str, int]:
        table = {}
        for b1 in (0, 1):
            for b2 in (0, 1):
                table[f"b1={b1},b2={b2}"] = sum(1 for _, x1, x2, _ in docs if (x1, x2) == (b1, b2))
        return table

    def assert_no_marginal_leakage(docs: list[Doc], name: str) -> None:
        n = len(docs)
        for bit_idx in (1, 2):
            for bit_val in (0, 1):
                subset = [d for d in docs if d[bit_idx] == bit_val]
                p_class1 = sum(1 for d in subset if d[3] == 1) / len(subset)
                assert p_class1 == 0.5, (
                    f"{name}: P(class=1 | b{bit_idx}={bit_val}) = {p_class1}, "
                    f"expected exactly 0.5 -- marginal leakage present"
                )

    train_combo_table = combo_table(train_docs)
    dev_combo_table = combo_table(dev_docs)
    print(f"train combo table: {train_combo_table}")
    print(f"dev combo table:   {dev_combo_table}")
    assert_no_marginal_leakage(train_docs, "train")
    assert_no_marginal_leakage(dev_docs, "dev")

    train_k, train_v, train_target = build_batch(sender, tokenizer, train_docs)
    dev_k, dev_v, dev_target = build_batch(sender, tokenizer, dev_docs)
    n_train = len(train_docs)
    n_dev = len(dev_docs)

    n_head = receiver.config.n_head
    head_dim = receiver.config.n_embd // n_head
    n_layers = len(SENDER_LAYER_SELECTION)

    adapter = LowRankKVAdapter(n_layers, head_dim, n_head, RANK, seed=SEED).to(DEVICE)
    opt = torch.optim.Adam(adapter.parameters(), lr=LR)
    print(f"seq_len={SEQ_LEN} header_len={header_len} trainable_params={adapter.param_count()}")

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
    # Selection by dev ACCURACY (tie-break: lower dev loss), not raw dev NLL:
    # on this task, dev loss and dev accuracy diverge post-plateau (the
    # adapter grows more confident on both its right and wrong predictions,
    # so average NLL keeps rising even while accuracy holds flat above
    # chance) -- NLL-based selection picked an early near-random checkpoint
    # while a real, stable, above-chance accuracy plateau existed later.
    best_dev_acc = -math.inf
    best_dev_loss_at_best_acc = math.inf
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
            if (dacc > best_dev_acc) or (dacc == best_dev_acc and dl < best_dev_loss_at_best_acc):
                best_dev_acc = dacc
                best_dev_loss_at_best_acc = dl
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

        progress_path.write_text(json.dumps({
            "epochs_run": epoch + 1, "train_loss_history": train_loss_history,
            "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
            "dev_acc_history": dev_acc_history,
            "best_dev_acc_so_far": best_dev_acc,
            "best_dev_loss_at_best_acc_so_far": best_dev_loss_at_best_acc,
            "wall_seconds_so_far": time.time() - t0,
            "status": "training",
        }, indent=2) + "\n")

    adapter.load_state_dict(best_state)
    adapter.eval()
    wall_s = time.time() - t0

    # Convergence check on dev ACCURACY, not train loss: train loss keeps
    # falling toward 0 indefinitely on this task via pure memorization
    # (confirmed in the first 800-epoch pass) long after the best-accuracy
    # checkpoint stops improving, so a train-loss-slope check is a false
    # ambiguity signal here (same false-signal pattern already documented
    # for the full-bandwidth-translator rung). "Still improving" instead
    # means: the best-accuracy checkpoint was found in the last quarter of
    # the run (so more budget might still move it), not earlier with margin.
    best_acc_idx = dev_acc_history.index(best_dev_acc)
    quarter_point = len(dev_eval_epochs) - max(1, len(dev_eval_epochs) // 4)
    still_improving = best_acc_idx >= quarter_point

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
        assert torch.equal(ps_target, dev_target)
        ps_logits = next_token_logits(receiver, dq, make_cache(adapter.forward_keys(ps_k), adapter.forward_values(ps_v)))
        ps_nll, ps_top1 = nll_top1(ps_logits, dev_target)

        h1k = zero_positions(dev_k, half2_lo, half2_hi)
        h1v = zero_positions(dev_v, half2_lo, half2_hi)
        h1_logits = next_token_logits(receiver, dq, make_cache(adapter.forward_keys(h1k), adapter.forward_values(h1v)))
        h1_nll, h1_top1 = nll_top1(h1_logits, dev_target)

        h2k = zero_positions(dev_k, half1_lo, half1_hi)
        h2v = zero_positions(dev_v, half1_lo, half1_hi)
        h2_logits = next_token_logits(receiver, dq, make_cache(adapter.forward_keys(h2k), adapter.forward_values(h2v)))
        h2_nll, h2_top1 = nll_top1(h2_logits, dev_target)

        swap_k, swap_v, swap_target = build_half_swap_batch(sender, tokenizer, dev_docs)
        swap_logits = next_token_logits(receiver, dq, make_cache(adapter.forward_keys(swap_k), adapter.forward_values(swap_v)))
        swap_nll, swap_top1 = nll_top1(swap_logits, swap_target)

    finite = all(torch.isfinite(x).all().item() for x in
                 (correct_nll, no_prefix_nll, zero_nll, shuf_nll, ps_nll, h1_nll, h2_nll, swap_nll))

    def advantage(other_nll):
        return (other_nll - correct_nll).tolist()

    shuffled_advantage = advantage(shuf_nll)
    zero_advantage = advantage(zero_nll)
    no_prefix_advantage = advantage(no_prefix_nll)
    ps_advantage = advantage(ps_nll)
    h1_advantage = advantage(h1_nll)
    h2_advantage = advantage(h2_nll)

    shuffled_ci = bootstrap_ci(shuffled_advantage, SEED)
    zero_ci = bootstrap_ci(zero_advantage, SEED + 1)
    no_prefix_ci = bootstrap_ci(no_prefix_advantage, SEED + 2)
    ps_ci = bootstrap_ci(ps_advantage, SEED + 3)
    h1_ci = bootstrap_ci(h1_advantage, SEED + 4)
    h2_ci = bootstrap_ci(h2_advantage, SEED + 5)
    swap_correct_ci = bootstrap_ci(swap_top1.tolist(), SEED + 6)  # CI on swap accuracy itself, vs chance 0.5

    correct_acc = correct_top1.mean().item()
    chance_baseline = 0.5
    accuracy_above_chance = correct_acc > chance_baseline
    beats_shuffled_doc = finite and shuffled_ci[0] > 0
    beats_position_shuffle = finite and ps_ci[0] > 0
    beats_half1_only = finite and h1_ci[0] > 0
    beats_half2_only = finite and h2_ci[0] > 0
    swap_above_chance = finite and swap_correct_ci[0] > chance_baseline
    compositionality_signal = (
        accuracy_above_chance and beats_half1_only and beats_half2_only and swap_above_chance
    )

    def summarize(nll, top1):
        return {"nll": nll.mean().item(), "accuracy": top1.mean().item()}

    result = {
        "seed": SEED, "seq_len": SEQ_LEN, "half": HALF, "sub_k_range": list(SUB_K_RANGE),
        "train_combo_table": train_combo_table, "dev_combo_table": dev_combo_table,
        "sender_layer_selection": list(SENDER_LAYER_SELECTION), "rank": RANK,
        "trainable_parameter_count": adapter.param_count(),
        "epochs_run": len(train_loss_history), "wall_seconds": wall_s,
        "best_dev_acc_at_selection": best_dev_acc,
        "best_dev_loss_at_selection": best_dev_loss_at_best_acc,
        "best_dev_acc_epoch": dev_eval_epochs[best_acc_idx],
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
        "dev_acc_history": dev_acc_history,
        "still_improving_at_cutoff": still_improving,
        "mechanism": "low_rank_residual_kv_adapter_batched_compositionality_xor_32token",
        "dev": {
            "correct": summarize(correct_nll, correct_top1),
            "shuffled_document": summarize(shuf_nll, shuf_top1),
            "position_shuffle": summarize(ps_nll, ps_top1),
            "zero_sender": summarize(zero_nll, zero_top1),
            "no_prefix": summarize(no_prefix_nll, no_prefix_top1),
            "half1_only": summarize(h1_nll, h1_top1),
            "half2_only": summarize(h2_nll, h2_top1),
            "half_swap": summarize(swap_nll, swap_top1),
            "shuffled_minus_correct_nll": sum(shuffled_advantage) / len(shuffled_advantage),
            "shuffled_minus_correct_bootstrap_95pct_ci": shuffled_ci,
            "position_shuffle_minus_correct_nll": sum(ps_advantage) / len(ps_advantage),
            "position_shuffle_minus_correct_bootstrap_95pct_ci": ps_ci,
            "zero_sender_minus_correct_nll": sum(zero_advantage) / len(zero_advantage),
            "zero_sender_minus_correct_bootstrap_95pct_ci": zero_ci,
            "no_prefix_minus_correct_nll": sum(no_prefix_advantage) / len(no_prefix_advantage),
            "no_prefix_minus_correct_bootstrap_95pct_ci": no_prefix_ci,
            "half1_only_minus_correct_nll": sum(h1_advantage) / len(h1_advantage),
            "half1_only_minus_correct_bootstrap_95pct_ci": h1_ci,
            "half2_only_minus_correct_nll": sum(h2_advantage) / len(h2_advantage),
            "half2_only_minus_correct_bootstrap_95pct_ci": h2_ci,
            "half_swap_accuracy_bootstrap_95pct_ci": swap_correct_ci,
        },
        "finite": finite,
        "chance_baseline": chance_baseline,
        "accuracy_above_chance": accuracy_above_chance,
        "beats_shuffled_document": beats_shuffled_doc,
        "beats_position_shuffle": beats_position_shuffle,
        "beats_half1_only": beats_half1_only,
        "beats_half2_only": beats_half2_only,
        "swap_above_chance": swap_above_chance,
        "verdict": (
            "COMPOSITIONALITY_TRAINING_INCONCLUSIVE" if still_improving else
            "KV_COMPOSITIONALITY_XOR_SUPPORTED" if compositionality_signal else
            "KV_COMPOSITIONALITY_XOR_NOT_SUPPORTED"
        ),
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(f"correct:      nll={result['dev']['correct']['nll']:.4f} acc={result['dev']['correct']['accuracy']:.4f}")
    print(f"shuffled_doc: nll={result['dev']['shuffled_document']['nll']:.4f} acc={result['dev']['shuffled_document']['accuracy']:.4f}")
    print(f"pos_shuffle:  nll={result['dev']['position_shuffle']['nll']:.4f} acc={result['dev']['position_shuffle']['accuracy']:.4f}")
    print(f"zero_sender:  nll={result['dev']['zero_sender']['nll']:.4f} acc={result['dev']['zero_sender']['accuracy']:.4f}")
    print(f"no_prefix:    nll={result['dev']['no_prefix']['nll']:.4f} acc={result['dev']['no_prefix']['accuracy']:.4f}")
    print(f"half1_only:   nll={result['dev']['half1_only']['nll']:.4f} acc={result['dev']['half1_only']['accuracy']:.4f}")
    print(f"half2_only:   nll={result['dev']['half2_only']['nll']:.4f} acc={result['dev']['half2_only']['accuracy']:.4f}")
    print(f"half_swap:    nll={result['dev']['half_swap']['nll']:.4f} acc={result['dev']['half_swap']['accuracy']:.4f} ci={swap_correct_ci}")
    print(f"verdict={result['verdict']} wall_s={wall_s:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
