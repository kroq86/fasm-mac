#!/usr/bin/env python3
"""Third minimum-alignment rung: low-rank cross-channel mixing, no bias.

z = U @ (V^T @ LayerNorm(h_sender)), U: (768, r), V: (768, r), b=0 by
construction. Diagonal (identity, scalar, and per-channel scale-only) was
shown insufficient for document-specific transfer at the fixed GPT-2 layer 6
-> DistilGPT2 block 3 site (see phase1_diagonal_scale_only_result.json:
correct vs shuffled CI [-0.004972, 0.003989], straddles zero). This rung's
only new degree of freedom relative to scale-only is cross-channel mixing;
b=0 deliberately, to avoid the additive-offset calibration confound found in
the scale+bias rung (zero-sender there closed 29 of 32.77 percentage points
of headroom on its own; here zero input still gives z=0 identically, so that
escape hatch is closed by construction, same as scale-only).

Minimal-cost falsification design, not a convergence study: for each rank in
RANKS (ascending), train with a fixed 30-epoch budget, dev scored only every
DEV_EVAL_EVERY epochs, no adaptive extension. Primary gate is correct vs
shuffled-document paired bootstrap CI excluding zero. Stop at the first rank
that passes; if none pass through the largest rank, report
LOW_RANK_NO_DOCUMENT_SPECIFIC_TRANSFER. Dense is not run by this script.

Reuses the identical document-selection/preparation logic, corpus, site, and
scoring window as phase1_scalar_sweep.py / phase1_diagonal_scale_only.py so
results are causally comparable across rungs -- only the bridge family (and
now its rank) changes. Test remains closed (fingerprinted, never tokenized
or scored).
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

SEED = 20260904
PREFIX = 64
SUFFIX = 160
SCORED = 32
TRAIN_DOCS = 64
DEV_DOCS = 32
SENDER_LAYER = 6
RECEIVER_BLOCK = 3
RANKS = (1, 2, 4, 8)
MAX_EPOCHS = 30
DEV_EVAL_EVERY = 5
STILL_IMPROVING_REL_THRESHOLD = 0.01
LR = 1e-2
V_INIT_STD = 0.02


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_documents(path: Path) -> list[str]:
    docs = [" ".join(x.split()) for x in path.read_text().split("<|endoftext|>")]
    docs = [x for x in docs if x]
    if len(docs) != len(set(docs)):
        raise RuntimeError("exact duplicate documents in source corpus")
    return docs


def bucket(doc: str) -> str:
    value = int(sha(doc.encode())[:8], 16) % 100
    return "train" if value < 70 else "dev" if value < 85 else "test"


def select(tokenizer, docs: list[str], split: str, count: int) -> list[dict]:
    chosen = []
    for doc in sorted((x for x in docs if bucket(x) == split), key=lambda x: sha(x.encode())):
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if len(ids) >= PREFIX + SUFFIX:
            chosen.append({"sha256": sha(doc.encode()), "ids": ids[: PREFIX + SUFFIX]})
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"only {len(chosen)} eligible {split} documents")
    return chosen


def prepare(sender, tokenizer, records: list[dict]) -> list[dict]:
    prepared = []
    with torch.no_grad():
        for record in records:
            tokens = record["ids"]
            suffix = torch.tensor([tokens[PREFIX:]], dtype=torch.long)
            receiver_ids = torch.cat((torch.tensor([[tokenizer.eos_token_id]]), suffix), dim=1)
            prefix = torch.tensor([tokens[:PREFIX]], dtype=torch.long)
            out = sender(prefix, use_cache=False, output_hidden_states=True)
            h = out.hidden_states[SENDER_LAYER][:, -1, :].float()
            h = torch.nn.functional.layer_norm(h, (h.shape[-1],)).detach()
            prepared.append({"sha256": record["sha256"], "ids": receiver_ids, "h": h})
    return prepared


class LowRank(nn.Module):
    """z = U @ (V^T @ h). U starts at zero (z=0 at init, safe warm start);
    V starts small-random so dL/dU is generically nonzero from step one, and
    dL/dV becomes live as soon as U moves off zero. No bias anywhere: with
    h=0, z=0 identically for any U, V -- the zero-sender calibration escape
    hatch found in scale+bias cannot occur here by construction."""

    def __init__(self, width: int, rank: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.U = nn.Parameter(torch.zeros(width, rank))
        self.V = nn.Parameter(torch.randn(width, rank, generator=g) * V_INIT_STD)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = h @ self.V          # (batch, rank)
        return w @ self.U.T     # (batch, width)

    def param_count(self) -> int:
        return self.U.numel() + self.V.numel()


def make_injected_forward(receiver, block_index: int):
    block = receiver.transformer.h[block_index]
    state = {"extra": None}

    def hook(_module, args):
        hidden = args[0].clone()
        hidden[:, 0, :] = hidden[:, 0, :] + state["extra"]
        return (hidden,) + args[1:]

    handle = block.register_forward_pre_hook(hook)

    def run(ids: torch.Tensor, extra_row0: torch.Tensor):
        state["extra"] = extra_row0
        return receiver(input_ids=ids, use_cache=False).logits

    return run, handle


def scored_ce(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    pred = logits[:, -SCORED - 1 : -1].float()
    target = ids[:, -SCORED:]
    losses = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none"
    )
    return losses.mean()


def per_doc_nll_top1(logits: torch.Tensor, ids: torch.Tensor) -> tuple[float, float]:
    pred = logits[:, -SCORED - 1 : -1].float()
    target = ids[:, -SCORED:]
    losses = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none"
    )
    top1 = (pred.argmax(-1) == target).float().mean().item()
    return losses.mean().item(), top1


def bootstrap_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return [means[249], means[9749]]


def run_rank(rank: int, receiver, train, dev, dev_records, full_rows, progress_path) -> dict:
    torch.manual_seed(SEED + rank)
    bridge = LowRank(768, rank, seed=SEED + rank)
    run, handle = make_injected_forward(receiver, RECEIVER_BLOCK)
    opt = torch.optim.Adam(bridge.parameters(), lr=LR)

    z0 = bridge(train[0]["h"])
    logits0 = run(train[0]["ids"], z0)
    loss0 = scored_ce(logits0, train[0]["ids"])
    loss0.backward()
    gU = bridge.U.grad
    if gU is None or not torch.isfinite(gU).all() or gU.abs().max().item() == 0.0:
        handle.remove()
        raise RuntimeError(f"rank={rank} initialization has no learning signal (U.grad)")
    opt.zero_grad()

    t0 = time.time()
    best_dev = math.inf
    best_state = {k: v.clone() for k, v in bridge.state_dict().items()}
    dev_history: list[float] = []
    dev_eval_epochs: list[int] = []
    train_loss_history: list[float] = []

    def dev_loss() -> float:
        bridge.eval()
        with torch.no_grad():
            total = sum(scored_ce(run(d["ids"], bridge(d["h"])), d["ids"]).item() for d in dev)
        bridge.train()
        return total / len(dev)

    for epoch in range(MAX_EPOCHS):
        bridge.train()
        epoch_losses = []
        for d in train:
            opt.zero_grad()
            z = bridge(d["h"])
            logits = run(d["ids"], z)
            loss = scored_ce(logits, d["ids"])
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        train_loss_history.append(sum(epoch_losses) / len(epoch_losses))

        is_last = epoch == MAX_EPOCHS - 1
        if (epoch + 1) % DEV_EVAL_EVERY == 0 or is_last:
            dl = dev_loss()
            dev_history.append(dl)
            dev_eval_epochs.append(epoch + 1)
            if dl < best_dev:
                best_dev = dl
                best_state = {k: v.clone() for k, v in bridge.state_dict().items()}

        progress_path.write_text(json.dumps({
            "rank": rank, "epochs_run": epoch + 1,
            "train_loss_history": train_loss_history,
            "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
            "best_dev_loss_so_far": best_dev, "wall_seconds_so_far": time.time() - t0,
            "status": "training",
        }, indent=2) + "\n")

    handle.remove()
    bridge.load_state_dict(best_state)
    bridge.eval()
    wall_s = time.time() - t0

    window = train_loss_history[-(DEV_EVAL_EVERY + 1):]
    still_improving = False
    if len(window) >= 2 and window[0] != 0:
        rel_decrease = (window[0] - window[-1]) / abs(window[0])
        still_improving = rel_decrease > STILL_IMPROVING_REL_THRESHOLD

    bridge_sha256 = sha(json.dumps(
        {k: v.tolist() for k, v in best_state.items()}, sort_keys=True
    ).encode())

    run2, handle2 = make_injected_forward(receiver, RECEIVER_BLOCK)
    n = len(dev)
    with torch.no_grad():
        neutral_rows, chosen_rows, shuffled_rows = [], [], []
        for i, d in enumerate(dev):
            zero = torch.zeros_like(d["h"])
            nll, top1 = per_doc_nll_top1(run2(d["ids"], zero), d["ids"])
            neutral_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

            z = bridge(d["h"])
            nll, top1 = per_doc_nll_top1(run2(d["ids"], z), d["ids"])
            chosen_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

            other_h = dev[(i - 1) % n]["h"]
            z_shuf = bridge(other_h)
            nll, top1 = per_doc_nll_top1(run2(d["ids"], z_shuf), d["ids"])
            shuffled_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})
    handle2.remove()

    gains = [a["nll"] - c["nll"] for a, c in zip(neutral_rows, chosen_rows)]
    content_advantage = [s["nll"] - c["nll"] for s, c in zip(shuffled_rows, chosen_rows)]
    headroom = [a["nll"] - f["nll"] for a, f in zip(neutral_rows, full_rows)]
    gain_mean = sum(gains) / len(gains)
    headroom_mean = sum(headroom) / len(headroom)
    fraction = gain_mean / headroom_mean if headroom_mean > 0 else float("nan")
    finite = all(
        math.isfinite(r["nll"]) and math.isfinite(r["top1"])
        for rows in (neutral_rows, chosen_rows, shuffled_rows) for r in rows
    )
    gain_ci = bootstrap_ci(gains, SEED)
    content_ci = bootstrap_ci(content_advantage, SEED + 1)
    gate_pass = finite and content_ci[0] > 0

    return {
        "rank": rank,
        "trainable_parameter_count": bridge.param_count(),
        "bridge_sha256": bridge_sha256,
        "epochs_run": len(train_loss_history),
        "wall_seconds": wall_s,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs,
        "dev_loss_history": dev_history,
        "still_improving_at_cutoff": still_improving,
        "dev": {
            "neutral": {"nll": sum(r["nll"] for r in neutral_rows) / n, "top1": sum(r["top1"] for r in neutral_rows) / n},
            "chosen_low_rank": {"nll": sum(r["nll"] for r in chosen_rows) / n, "top1": sum(r["top1"] for r in chosen_rows) / n},
            "shuffled_document": {"nll": sum(r["nll"] for r in shuffled_rows) / n, "top1": sum(r["top1"] for r in shuffled_rows) / n},
            "receiver_matched_headroom_nll": headroom_mean,
            "fraction_of_receiver_matched_headroom_closed": fraction,
            "neutral_minus_chosen_nll": gain_mean,
            "neutral_minus_chosen_bootstrap_95pct_ci": gain_ci,
            "shuffled_minus_chosen_nll": sum(content_advantage) / len(content_advantage),
            "shuffled_minus_chosen_bootstrap_95pct_ci": content_ci,
        },
        "finite": finite,
        "gate_pass_correct_beats_shuffled": gate_pass,
        "per_document_dev": {"neutral": neutral_rows, "chosen_low_rank": chosen_rows, "shuffled_document": shuffled_rows},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval()
    for m in (sender, receiver):
        for p in m.parameters():
            p.requires_grad_(False)

    docs = load_documents(args.corpus)
    corpus_sha256 = sha(args.corpus.read_bytes())
    train_records = select(tokenizer, docs, "train", TRAIN_DOCS)
    dev_records = select(tokenizer, docs, "dev", DEV_DOCS)
    train = prepare(sender, tokenizer, train_records)
    dev = prepare(sender, tokenizer, dev_records)

    with torch.no_grad():
        full_rows = []
        for d, rec in zip(dev, dev_records):
            full_ids = torch.tensor([rec["ids"]], dtype=torch.long)
            logits = receiver(full_ids, use_cache=False).logits
            nll, top1 = per_doc_nll_top1(logits, full_ids)
            full_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    per_rank = []
    stopped_at_rank = None
    for rank in RANKS:
        result = run_rank(rank, receiver, train, dev, dev_records, full_rows, progress_path)
        per_rank.append(result)
        args.output.write_text(json.dumps({
            "seed": SEED, "corpus_sha256": corpus_sha256, "test_tokenized_or_scored": False,
            "ranks_planned": list(RANKS), "ranks_run": [r["rank"] for r in per_rank],
            "per_rank": per_rank, "stopped_at_rank": stopped_at_rank,
            "verdict": "IN_PROGRESS",
        }, sort_keys=True, indent=2) + "\n")
        if result["gate_pass_correct_beats_shuffled"]:
            stopped_at_rank = rank
            break

    verdict = "LOW_RANK_DOCUMENT_SPECIFIC_TRANSFER" if stopped_at_rank is not None else "LOW_RANK_NO_DOCUMENT_SPECIFIC_TRANSFER"
    final = {
        "seed": SEED, "corpus_sha256": corpus_sha256, "test_tokenized_or_scored": False,
        "sender_layer": SENDER_LAYER, "receiver_block": RECEIVER_BLOCK,
        "bridge": "low_rank_no_bias_no_mixing_with_diagonal",
        "ranks_planned": list(RANKS), "ranks_run": [r["rank"] for r in per_rank],
        "per_rank": per_rank, "stopped_at_rank": stopped_at_rank,
        "verdict": verdict,
    }
    args.output.write_text(json.dumps(final, sort_keys=True, indent=2) + "\n")
    print(f"ranks_run={[r['rank'] for r in per_rank]} stopped_at_rank={stopped_at_rank} verdict={verdict}")
    for r in per_rank:
        d = r["dev"]
        print(f"  rank={r['rank']} gain={d['neutral_minus_chosen_nll']:.6f} "
              f"shuffled_minus_chosen={d['shuffled_minus_chosen_nll']:.6f} "
              f"ci={d['shuffled_minus_chosen_bootstrap_95pct_ci']} "
              f"fraction={d['fraction_of_receiver_matched_headroom_closed']:.6f} "
              f"gate_pass={r['gate_pass_correct_beats_shuffled']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
