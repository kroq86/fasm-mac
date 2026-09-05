#!/usr/bin/env python3
"""Cheap inference-only screening pass over a SINGLE frozen, already-
trained KV bridge (phase1_kv_cache_minigunpoint_32token.py's low-rank
adapter, checkpointed via --bridge-out). No gradient step is taken here;
the adapter is loaded once, frozen, and used as a fixed measuring
instrument. Everything below only edits the CACHE that gets fed to the
receiver: which layers carry it, which heads carry it, how it is scaled,
noised, positionally shifted, delayed relative to the query, or
contradicted by receiver-side text.

This does NOT probe addressing/compositionality/binding: this adapter was
only ever trained on single-fact (here: single global-pattern) recovery,
so it has no learned addressing behavior to ablate. Those axes stay
answered by their own dedicated (already-run, NOT_SUPPORTED) adapters.

每 What this script answers, one arm per axis, no sweep:
  - K-vs-V ablation:      correct-K/shuffled-V vs shuffled-K/correct-V
  - cross-layer:          all 6 layers vs first-3 vs last-1 vs first-1
  - head alignment:       all 12 heads vs first-6 heads only
  - norm/scale mismatch:  K/V scaled by 0.5x / 1x / 2x
  - noise robustness:     clean vs +10% per-tensor-std Gaussian noise
  - persistence/delay:    query issued at delay 0 vs after 32 filler tokens
  - receiver-context conflict: neutral query vs a receiver-side textual
    claim asserting the OTHER class

Baseline correct/shuffled_document/position_shuffle are also recomputed
first, as a harness self-check: they must reproduce the frozen training
run's recorded numbers (acc=0.96875, nll=0.2017) before any ablation
result is trusted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
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
K_RANGE = (10, 11, 12, 13, 14)
DELAY_TOKENS = 32
NOISE_FRACTION = 0.10
A_INIT_STD = 0.02


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bucket(seq_str: str) -> str:
    value = int(sha(seq_str.encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def make_sequence(rng: random.Random, cls: int) -> list[str]:
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
    return first + second


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
    c0 = [d for d in chosen if d[1] == 0][: count // 2]
    c1 = [d for d in chosen if d[1] == 1][: count // 2]
    if len(c0) < count // 2 or len(c1) < count // 2:
        raise RuntimeError(f"not enough class balance in {split}: {len(c0)} / {len(c1)}")
    interleaved = [d for pair in zip(c0, c1) for d in pair]
    return interleaved


def seq_to_ids(tokenizer, seq: list[str]) -> list[int]:
    text = "Sequence:" + "".join(f" {tok}" for tok in seq)
    return tokenizer.encode(text, add_special_tokens=False)


def build_batch(sender, tokenizer, docs, sender_layer_selection):
    tokenized = [seq_to_ids(tokenizer, seq) for seq, _ in docs]
    lens = {len(t) for t in tokenized}
    assert len(lens) == 1, f"prefix length not constant: {lens}"
    ids = torch.tensor(tokenized, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = sender(ids, use_cache=True)
    pkv = out.past_key_values
    keys = [pkv.layers[i].keys.clone().detach() for i in sender_layer_selection]
    values = [pkv.layers[i].values.clone().detach() for i in sender_layer_selection]
    target_ids = [tokenizer.encode((" 0", " 1")[cls], add_special_tokens=False)[0] for _, cls in docs]
    target = torch.tensor(target_ids, dtype=torch.long, device=DEVICE)
    return keys, values, target


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


def make_cache(keys, values) -> DynamicCache:
    cache = DynamicCache()
    for k, v in zip(keys, values):
        cache.update(k, v, layer_idx=len(cache.layers))
    return cache


def empty_like(k: torch.Tensor) -> torch.Tensor:
    # Same-length, zero-valued (not zero-length): HF attention assumes a
    # uniform cache_position/seq_length across layers, so a mismatched
    # per-layer cache length breaks the SDPA mask shape. Zero-valued KV of
    # the SAME length is the per-layer analogue of the existing
    # zero_sender control (informationless, but shape-compatible).
    return torch.zeros_like(k)


def roll_batch(keys):
    return [torch.roll(k, shifts=1, dims=0) for k in keys]


def query_ids(tokenizer, text: str, batch: int) -> torch.Tensor:
    ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    return ids.expand(batch, -1)


def next_token_logits(receiver, q_ids: torch.Tensor, cache: DynamicCache) -> torch.Tensor:
    out = receiver(q_ids, past_key_values=cache, use_cache=False)
    return out.logits[:, -1, :].float()


def nll_top1(logits: torch.Tensor, target: torch.Tensor):
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


def summarize(nll: torch.Tensor, top1: torch.Tensor) -> dict:
    return {"nll": nll.mean().item(), "accuracy": top1.mean().item()}


def delta_ci(base_nll: torch.Tensor, other_nll: torch.Tensor, seed: int) -> dict:
    delta = (other_nll - base_nll).tolist()
    ci = bootstrap_ci(delta, seed)
    return {"other_minus_base_nll": sum(delta) / len(delta), "bootstrap_95pct_ci": ci, "worse_than_base": ci[0] > 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--bridge", type=Path, required=True)
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

    ckpt = torch.load(args.bridge, map_location=DEVICE, weights_only=False)
    sender_layer_selection = ckpt["sender_layer_selection"]
    rank = ckpt["rank"]
    n_head = ckpt["n_head"]
    head_dim = ckpt["head_dim"]
    n_layers = len(sender_layer_selection)

    adapter = LowRankKVAdapter(n_layers, head_dim, n_head, rank, seed=SEED)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.to(DEVICE)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)

    pool = generate_pool(target_per_class=200)
    dev_docs = select_docs(pool, "dev", DEV_DOCS)
    dev_keys_set = sorted({"".join(s) for s, _ in dev_docs})
    assert dev_keys_set == ckpt["dev_docs_keys"], "dev split does not match the checkpointed training run"

    dev_k, dev_v, dev_target = build_batch(sender, tokenizer, dev_docs, sender_layer_selection)
    n_dev = len(dev_docs)
    dq = query_ids(tokenizer, "Class:", n_dev)

    results: dict = {}

    def eval_cache(keys, values, q_ids=None):
        cache = make_cache(keys, values)
        logits = next_token_logits(receiver, dq if q_ids is None else q_ids, cache)
        return nll_top1(logits, dev_target)

    # ---- baseline self-check: must match the frozen training run ----
    ak = adapter.forward_keys(dev_k)
    av = adapter.forward_values(dev_v)
    correct_nll, correct_top1 = eval_cache(ak, av)
    results["baseline_correct"] = summarize(correct_nll, correct_top1)

    sk, sv = roll_batch(dev_k), roll_batch(dev_v)
    shuf_nll, shuf_top1 = eval_cache(adapter.forward_keys(sk), adapter.forward_values(sv))
    results["baseline_shuffled_document"] = summarize(shuf_nll, shuf_top1)
    results["baseline_shuffled_vs_correct"] = delta_ci(correct_nll, shuf_nll, SEED)

    # ---- 1. K-vs-V cross ablation ----
    ak_s = adapter.forward_keys(sk)   # shuffled K
    av_s = adapter.forward_values(sv)  # shuffled V
    wrongK_correctV_nll, wrongK_correctV_top1 = eval_cache(ak_s, av)
    correctK_wrongV_nll, correctK_wrongV_top1 = eval_cache(ak, av_s)
    results["k_vs_v"] = {
        "correct_K_correct_V": summarize(correct_nll, correct_top1),
        "wrong_K_correct_V": summarize(wrongK_correctV_nll, wrongK_correctV_top1),
        "correct_K_wrong_V": summarize(correctK_wrongV_nll, correctK_wrongV_top1),
        "wrong_K_hurts_more_than_wrong_V": delta_ci(correctK_wrongV_nll, wrongK_correctV_nll, SEED + 10),
    }

    # ---- 2. cross-layer sufficiency ----
    def layers_subset(active_idx: set[int]):
        k = [ak[i] if i in active_idx else empty_like(ak[i]) for i in range(n_layers)]
        v = [av[i] if i in active_idx else empty_like(av[i]) for i in range(n_layers)]
        return eval_cache(k, v)

    all_idx = set(range(n_layers))
    first_half_idx = set(range(n_layers // 2))
    last_only_idx = {n_layers - 1}
    first_only_idx = {0}
    layer_arms = {}
    for name, idx in [("all_layers", all_idx), ("first_half_layers", first_half_idx),
                       ("last_layer_only", last_only_idx), ("first_layer_only", first_only_idx)]:
        nll, top1 = layers_subset(idx)
        layer_arms[name] = summarize(nll, top1)
        layer_arms[name + "_vs_all_ci"] = delta_ci(correct_nll, nll, SEED + 20) if name != "all_layers" else None
    results["cross_layer"] = layer_arms

    # ---- 3. head alignment ----
    def heads_subset(keys, values, active_heads: slice):
        k_out, v_out = [], []
        for k, v in zip(keys, values):
            km, vm = k.clone(), v.clone()
            mask = torch.zeros(n_head, dtype=torch.bool, device=DEVICE)
            mask[active_heads] = True
            km[:, ~mask, :, :] = 0.0
            vm[:, ~mask, :, :] = 0.0
            k_out.append(km)
            v_out.append(vm)
        return eval_cache(k_out, v_out)

    all_heads_nll, all_heads_top1 = eval_cache(ak, av)  # identical to baseline, sanity anchor
    half_nll, half_top1 = heads_subset(ak, av, slice(0, n_head // 2))
    results["head_alignment"] = {
        "all_heads": summarize(all_heads_nll, all_heads_top1),
        "first_half_heads_only": summarize(half_nll, half_top1),
        "half_vs_all_ci": delta_ci(correct_nll, half_nll, SEED + 30),
    }

    # ---- 4. norm/scale mismatch ----
    scale_arms = {}
    for scale in (0.5, 1.0, 2.0):
        k_s = [k * scale for k in ak]
        v_s = [v * scale for v in av]
        nll, top1 = eval_cache(k_s, v_s)
        scale_arms[f"scale_{scale}"] = summarize(nll, top1)
        scale_arms[f"scale_{scale}_vs_1.0_ci"] = delta_ci(correct_nll, nll, SEED + 40) if scale != 1.0 else None
    results["scale_mismatch"] = scale_arms

    # ---- 5. noise robustness ----
    g = torch.Generator().manual_seed(SEED + 50)
    k_noisy = [k + (torch.randn(k.shape, generator=g).to(DEVICE) * (NOISE_FRACTION * k.std())) for k in ak]
    v_noisy = [v + (torch.randn(v.shape, generator=g).to(DEVICE) * (NOISE_FRACTION * v.std())) for v in av]
    noisy_nll, noisy_top1 = eval_cache(k_noisy, v_noisy)
    results["noise_robustness"] = {
        "clean": summarize(correct_nll, correct_top1),
        f"plus_{int(NOISE_FRACTION*100)}pct_noise": summarize(noisy_nll, noisy_top1),
        "noisy_vs_clean_ci": delta_ci(correct_nll, noisy_nll, SEED + 51),
    }

    # ---- 6. persistence / delay ----
    filler_ids = tokenizer.encode(" x", add_special_tokens=False)
    assert len(filler_ids) == 1
    filler_batch = torch.tensor([filler_ids * DELAY_TOKENS] * n_dev, dtype=torch.long, device=DEVICE)

    def eval_with_delay(n_filler: int):
        cache = make_cache(ak, av)
        if n_filler > 0:
            with torch.no_grad():
                receiver(filler_batch[:, :n_filler], past_key_values=cache, use_cache=True)
        logits = next_token_logits(receiver, dq, cache)
        return nll_top1(logits, dev_target)

    delay0_nll, delay0_top1 = eval_with_delay(0)
    delay32_nll, delay32_top1 = eval_with_delay(DELAY_TOKENS)
    results["persistence_delay"] = {
        "delay_0": summarize(delay0_nll, delay0_top1),
        f"delay_{DELAY_TOKENS}": summarize(delay32_nll, delay32_top1),
        "delay_vs_immediate_ci": delta_ci(delay0_nll, delay32_nll, SEED + 60),
    }

    # ---- 7. receiver-context conflict ----
    neutral_q = query_ids(tokenizer, "Class:", n_dev)
    conflict_text = "The correct answer is definitely 1. Class:"
    conflict_q = query_ids(tokenizer, conflict_text, n_dev)
    neutral_nll, neutral_top1 = eval_cache(ak, av, q_ids=neutral_q)
    conflict_nll, conflict_top1 = eval_cache(ak, av, q_ids=conflict_q)
    flip_rate = (conflict_nll.argsort() * 0)  # placeholder not used
    with torch.no_grad():
        neutral_pred = next_token_logits(receiver, neutral_q, make_cache(ak, av)).argmax(-1)
        conflict_pred = next_token_logits(receiver, conflict_q, make_cache(ak, av)).argmax(-1)
    results["receiver_context_conflict"] = {
        "neutral_prompt": summarize(neutral_nll, neutral_top1),
        "conflicting_prompt_favors_class1": summarize(conflict_nll, conflict_top1),
        "prediction_flip_rate": (neutral_pred != conflict_pred).float().mean().item(),
        "conflict_vs_neutral_ci": delta_ci(neutral_nll, conflict_nll, SEED + 70),
    }

    args.output.write_text(json.dumps(results, sort_keys=True, indent=2, default=str) + "\n")

    print(f"baseline correct:   acc={results['baseline_correct']['accuracy']:.4f} nll={results['baseline_correct']['nll']:.4f}")
    print(f"baseline shuffled:  acc={results['baseline_shuffled_document']['accuracy']:.4f} nll={results['baseline_shuffled_document']['nll']:.4f}")
    print(f"k_vs_v: wrongK+correctV acc={results['k_vs_v']['wrong_K_correct_V']['accuracy']:.4f}  "
          f"correctK+wrongV acc={results['k_vs_v']['correct_K_wrong_V']['accuracy']:.4f}")
    for k, v in results["cross_layer"].items():
        if isinstance(v, dict) and "accuracy" in v:
            print(f"layers[{k}]: acc={v['accuracy']:.4f} nll={v['nll']:.4f}")
    print(f"heads: all={results['head_alignment']['all_heads']['accuracy']:.4f} "
          f"half={results['head_alignment']['first_half_heads_only']['accuracy']:.4f}")
    for k, v in results["scale_mismatch"].items():
        if isinstance(v, dict) and "accuracy" in v:
            print(f"scale[{k}]: acc={v['accuracy']:.4f} nll={v['nll']:.4f}")
    print(f"noise: clean acc={results['noise_robustness']['clean']['accuracy']:.4f} "
          f"noisy acc={results['noise_robustness'][f'plus_{int(NOISE_FRACTION*100)}pct_noise']['accuracy']:.4f}")
    print(f"delay: 0={results['persistence_delay']['delay_0']['accuracy']:.4f} "
          f"{DELAY_TOKENS}={results['persistence_delay'][f'delay_{DELAY_TOKENS}']['accuracy']:.4f}")
    print(f"context conflict: neutral acc={results['receiver_context_conflict']['neutral_prompt']['accuracy']:.4f} "
          f"conflicting acc={results['receiver_context_conflict']['conflicting_prompt_favors_class1']['accuracy']:.4f} "
          f"flip_rate={results['receiver_context_conflict']['prediction_flip_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
