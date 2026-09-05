#!/usr/bin/env python3
"""Second minimum-alignment rung: per-channel scale only, no bias, no
channel mixing. z_i = a_i * LayerNorm(h_sender)_i, 768 trainable
parameters (a), b fixed at 0.

Isolated from bias/diagonal-affine deliberately -- see the phase1 audit:
combining per-channel scale and bias in one rung would conflate two
different hypotheses (does each channel need its own GAIN, vs does each
channel need its own OFFSET). This rung answers only the first question.

Reuses the identical document-selection/preparation logic and protocol as
phase1_scalar_sweep.py (same corpus, same bucket()/select() functions,
same 64 train / 32 dev documents, same GPT-2 layer 6 -> DistilGPT2 block 3
site, same shuffled-document control, same scoring window, same bootstrap
style) so results are causally comparable to the scalar rung -- only the
bridge family changes. Test remains closed (fingerprinted, never
tokenized or scored).

a_i=0 for all i is a SAFE zero-init here (unlike the earlier dense
W=b=alpha=0 dead point): there is only one multiplicative layer
(a_i * h_i), so d(loss)/d(a_i) = downstream_grad_i * h_i, which is
generically non-zero at a=0 as long as h itself is non-zero. A first-step
gradient check is still included, matching the project's established
fail-closed discipline rather than assuming it.
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
MAX_EPOCHS = 30
DEV_EVAL_EVERY = 5  # dev is 32 forward passes; only worth paying for at sparse checkpoints
STILL_IMPROVING_REL_THRESHOLD = 0.01  # >1% relative train-loss drop over the last window => INCONCLUSIVE, not extended
LR = 1e-2


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


class DiagonalScaleOnly(nn.Module):
    def __init__(self, width: int = 768):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(width))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.a * h

    def param_count(self) -> int:
        return self.a.numel()


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bridge-out", type=Path, required=True)
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

    bridge = DiagonalScaleOnly(768)
    assert bridge.param_count() == 768
    run, handle = make_injected_forward(receiver, RECEIVER_BLOCK)
    opt = torch.optim.Adam(bridge.parameters(), lr=LR)

    # first-step gradient check: fail closed rather than assume safety
    z0 = bridge(train[0]["h"])
    logits0 = run(train[0]["ids"], z0)
    loss0 = scored_ce(logits0, train[0]["ids"])
    loss0.backward()
    g = bridge.a.grad
    if g is None or not torch.isfinite(g).all() or g.abs().max().item() == 0.0:
        raise RuntimeError("diagonal-scale-only initialization has no learning signal")
    opt.zero_grad()

    t0 = time.time()
    best_dev = math.inf
    best_state = {k: v.clone() for k, v in bridge.state_dict().items()}
    dev_history = []  # sparse: only epochs where dev was actually evaluated
    dev_eval_epochs = []
    train_loss_history = []  # dense: every epoch, cheap (already computed during the step)

    def dev_loss() -> float:
        bridge.eval()
        with torch.no_grad():
            total = sum(scored_ce(run(d["ids"], bridge(d["h"])), d["ids"]).item() for d in dev)
        bridge.train()
        return total / len(dev)

    # Minimal-cost falsification budget, not a convergence study: fixed 30
    # epochs, no patience-based extension, dev scored only every
    # DEV_EVAL_EVERY epochs (32 extra forward passes each time it runs).
    # If train loss is still clearly dropping at the end, that is reported
    # as its own verdict (TRAINING_INCONCLUSIVE) rather than used as a
    # reason to run longer.
    progress_path = args.output.with_name(args.output.stem + "_progress.json")
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
                torch.save(best_state, args.bridge_out)  # persist immediately, not only at the end

        progress_path.write_text(json.dumps({
            "epochs_run": epoch + 1,
            "train_loss_history": train_loss_history,
            "dev_eval_epochs": dev_eval_epochs,
            "dev_loss_history": dev_history,
            "best_dev_loss_so_far": best_dev,
            "wall_seconds_so_far": time.time() - t0,
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

    torch.save(bridge.state_dict(), args.bridge_out)
    progress_path.write_text(json.dumps({
        "epochs_run": len(train_loss_history),
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs,
        "dev_loss_history": dev_history,
        "best_dev_loss_so_far": best_dev,
        "wall_seconds_so_far": wall_s,
        "still_improving_at_cutoff": still_improving,
        "status": "training_done_computing_final_arms",
    }, indent=2) + "\n")
    bridge_sha256 = sha(args.bridge_out.read_bytes())

    # Final dev arms, same protocol as scalar rung: neutral (no injection),
    # chosen (trained diagonal-scale), shuffled-document control.
    run2, handle2 = make_injected_forward(receiver, RECEIVER_BLOCK)
    with torch.no_grad():
        neutral_rows, chosen_rows, shuffled_rows = [], [], []
        n = len(dev)
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

    # receiver-matched ceiling (per the corrected metric definition):
    # DistilGPT2 itself on the real full prefix+suffix.
    with torch.no_grad():
        full_rows = []
        for d, rec in zip(dev, dev_records):
            full_ids = torch.tensor([rec["ids"]], dtype=torch.long)
            logits = receiver(full_ids, use_cache=False).logits
            nll, top1 = per_doc_nll_top1(logits, full_ids)
            full_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

    gains = [n["nll"] - c["nll"] for n, c in zip(neutral_rows, chosen_rows)]
    content_advantage = [s["nll"] - c["nll"] for s, c in zip(shuffled_rows, chosen_rows)]
    headroom = [n["nll"] - f["nll"] for n, f in zip(neutral_rows, full_rows)]
    gain_mean = sum(gains) / len(gains)
    headroom_mean = sum(headroom) / len(headroom)
    fraction = gain_mean / headroom_mean if headroom_mean > 0 else float("nan")

    finite = all(
        math.isfinite(r["nll"]) and math.isfinite(r["top1"])
        for rows in (neutral_rows, chosen_rows, shuffled_rows, full_rows) for r in rows
    )
    gain_ci = bootstrap_ci(gains, SEED)
    content_ci = bootstrap_ci(content_advantage, SEED + 1)
    signal = finite and gain_ci[0] > 0 and content_ci[0] > 0

    result = {
        "seed": SEED,
        "corpus_sha256": corpus_sha256,
        "test_tokenized_or_scored": False,
        "train_documents": TRAIN_DOCS,
        "dev_documents": DEV_DOCS,
        "sender_layer": SENDER_LAYER,
        "receiver_block": RECEIVER_BLOCK,
        "bridge": "diagonal_scale_only_no_bias_no_mixing",
        "trainable_parameter_count": bridge.param_count(),
        "bridge_sha256": bridge_sha256,
        "epochs_run": len(train_loss_history),
        "wall_seconds": wall_s,
        "best_dev_loss_at_selection": best_dev,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs,
        "dev_loss_history": dev_history,
        "still_improving_at_cutoff": still_improving,
        "still_improving_rel_threshold": STILL_IMPROVING_REL_THRESHOLD,
        "dev": {
            "neutral": {"nll": sum(r["nll"] for r in neutral_rows) / n, "top1": sum(r["top1"] for r in neutral_rows) / n},
            "chosen_diagonal": {"nll": sum(r["nll"] for r in chosen_rows) / n, "top1": sum(r["top1"] for r in chosen_rows) / n},
            "shuffled_document": {"nll": sum(r["nll"] for r in shuffled_rows) / n, "top1": sum(r["top1"] for r in shuffled_rows) / n},
            "receiver_full_context_ceiling": {"nll": sum(r["nll"] for r in full_rows) / n, "top1": sum(r["top1"] for r in full_rows) / n},
            "receiver_matched_headroom_nll": headroom_mean,
            "fraction_of_receiver_matched_headroom_closed": fraction,
            "neutral_minus_chosen_nll": gain_mean,
            "neutral_minus_chosen_bootstrap_95pct_ci": gain_ci,
            "shuffled_minus_chosen_nll": sum(content_advantage) / len(content_advantage),
            "shuffled_minus_chosen_bootstrap_95pct_ci": content_ci,
        },
        "finite": finite,
        "verdict": (
            "DIAGONAL_SCALE_TRAINING_INCONCLUSIVE" if still_improving else
            "DIAGONAL_SCALE_SIGNAL" if signal else
            "DIAGONAL_SCALE_NO_SIGNAL"
        ),
        "per_document_dev": {
            "neutral": neutral_rows, "chosen_diagonal": chosen_rows,
            "shuffled_document": shuffled_rows, "receiver_full_context_ceiling": full_rows,
        },
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(encoded[:2000])
    print("...")
    print(f"gain={gain_mean:.6f} ci={gain_ci} headroom={headroom_mean:.6f} fraction={fraction:.6f} verdict={result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
