#!/usr/bin/env python3
"""Norm-calibrated rerun of phase1_latent_prefix_tokens.py, not a new
bridge family or site -- exactly one change from the uncalibrated run.

The uncalibrated run (phase1_latent_prefix_tokens_result.json) did not
cleanly test "does prefix-token fusion carry document-specific content".
Its trained projector collapsed receiver performance catastrophically
(no_prefix NLL 2.626 -> correct NLL 5.005, top-1 0.408 -> 0.155), while
zero_sender (8 literal zero vectors prepended, no training) stayed normal
(NLL 2.669). That isolates the cause to the *trained projector's output
magnitude/distribution*, not the prefix-slot mechanism itself: an
unconstrained linear map has no reason to produce vectors on the same norm
scale as DistilGPT2's real token embeddings, and StateBridge's own ablation
(arXiv:2608.13317) shows removing its norm-calibration step measurably
hurts performance for exactly this reason. That run is preserved as its own
artifact and its own verdict, `UNCONSTRAINED_LATENT_PREFIX_DESTABILIZES_RECEIVER`,
not silently overwritten or treated as inconvenient.

This rerun changes exactly one thing: each of the K projected latent
vectors is rescaled (no learned parameters involved in the rescale itself)
to match the median L2 norm of DistilGPT2's real token embedding rows
(`receiver.transformer.wte.weight`), before being prepended. No bias, no
nonlinearity, no new bridge family, no new site, no new data. A is
initialized small-random (not exactly zero) so the norm-rescale's
division is well-defined from the first step; with sender input forced to
zero, the projector output is exactly zero regardless of A, and the
rescale of an exact zero vector stays zero (0/eps*target=0) -- the
zero-sender invariant established throughout this project's no-bias rungs
is preserved.

Same four arms, same gate: primary correct-vs-shuffled paired
document-bootstrap CI > 0; secondary correct-vs-no_prefix. If the
projector again destabilizes the receiver, or correct~=shuffled even once
stable, the latent-prefix branch is closed for good under this protocol.
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
SENDER_LAYER = 12  # GPT-2's last block output, matching the site spike
NUM_LATENT_TOKENS = 8
MAX_EPOCHS = 30
DEV_EVAL_EVERY = 5
STILL_IMPROVING_REL_THRESHOLD = 0.01
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
            prefix_ids = torch.tensor([tokens[:PREFIX]], dtype=torch.long)
            suffix_ids = torch.tensor([tokens[PREFIX:]], dtype=torch.long)
            out = sender(prefix_ids, use_cache=False, output_hidden_states=True)
            H = out.hidden_states[SENDER_LAYER].float()  # (1, PREFIX, 768)
            H = torch.nn.functional.layer_norm(H, (H.shape[-1],)).detach()
            prepared.append({"sha256": record["sha256"], "suffix_ids": suffix_ids, "H": H})
    return prepared


class PrefixProjector(nn.Module):
    """latent_raw = A @ H, A: (K, PREFIX), no bias. Then each of the K
    output vectors is rescaled (no learned params in the rescale) to the
    fixed target_norm (median L2 norm of the receiver's real token
    embeddings), computed once from receiver.transformer.wte.weight before
    training and passed in un-trainable.

    A is initialized small-random, not exactly zero: the rescale divides by
    ||latent_raw||, which is singular at exactly zero. With H=0 (zero
    sender), latent_raw=0 for any A regardless of init, and the rescale of
    an exact zero vector is defined to stay zero (clamped denominator) --
    the zero-sender invariant is preserved by the H=0 case, not by A's
    init, so the small-random A init does not reopen the earlier
    dead-init class of bug."""

    def __init__(self, k: int, prefix: int, target_norm: float, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter(torch.randn(k, prefix, generator=g) * 0.02)
        self.target_norm = target_norm

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (1, PREFIX, width) -> (K, width)
        raw = self.A @ H.squeeze(0)
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return raw / norm * self.target_norm

    def param_count(self) -> int:
        return self.A.numel()


def receiver_logits(receiver, wte, suffix_ids: torch.Tensor, latent: torch.Tensor | None):
    suffix_embeds = wte(suffix_ids)  # (1, SUFFIX, width)
    if latent is None:
        combined = suffix_embeds
    else:
        combined = torch.cat([latent.unsqueeze(0), suffix_embeds], dim=1)
    return receiver(inputs_embeds=combined, use_cache=False).logits


def scored_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = logits[:, -SCORED - 1 : -1].float()
    losses = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none"
    )
    return losses.mean()


def per_doc_nll_top1(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred = logits[:, -SCORED - 1 : -1].float()
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
    ap.add_argument("--ceiling", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval()
    for m in (sender, receiver):
        for p in m.parameters():
            p.requires_grad_(False)
    wte = receiver.transformer.wte

    docs = load_documents(args.corpus)
    corpus_sha256 = sha(args.corpus.read_bytes())
    train_records = select(tokenizer, docs, "train", TRAIN_DOCS)
    dev_records = select(tokenizer, docs, "dev", DEV_DOCS)
    train = prepare(sender, tokenizer, train_records)
    dev = prepare(sender, tokenizer, dev_records)

    with torch.no_grad():
        embedding_norms = wte.weight.float().norm(dim=-1)
        target_norm = embedding_norms.median().item()

    bridge = PrefixProjector(NUM_LATENT_TOKENS, PREFIX, target_norm=target_norm, seed=SEED)
    opt = torch.optim.Adam(bridge.parameters(), lr=LR)

    z0 = bridge(train[0]["H"])
    logits0 = receiver_logits(receiver, wte, train[0]["suffix_ids"], z0)
    target0 = train[0]["suffix_ids"][:, -SCORED:]
    loss0 = scored_ce(logits0, target0)
    loss0.backward()
    g = bridge.A.grad
    if g is None or not torch.isfinite(g).all() or g.abs().max().item() == 0.0:
        raise RuntimeError("prefix-projector initialization has no learning signal")
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
            total = 0.0
            for d in dev:
                z = bridge(d["H"])
                logits = receiver_logits(receiver, wte, d["suffix_ids"], z)
                target = d["suffix_ids"][:, -SCORED:]
                total += scored_ce(logits, target).item()
        bridge.train()
        return total / len(dev)

    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    for epoch in range(MAX_EPOCHS):
        bridge.train()
        epoch_losses = []
        for d in train:
            opt.zero_grad()
            z = bridge(d["H"])
            logits = receiver_logits(receiver, wte, d["suffix_ids"], z)
            target = d["suffix_ids"][:, -SCORED:]
            loss = scored_ce(logits, target)
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
            "epochs_run": epoch + 1, "train_loss_history": train_loss_history,
            "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
            "best_dev_loss_so_far": best_dev, "wall_seconds_so_far": time.time() - t0,
            "status": "training",
        }, indent=2) + "\n")

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

    n = len(dev)
    with torch.no_grad():
        no_prefix_rows, zero_sender_rows, chosen_rows, shuffled_rows = [], [], [], []
        for i, d in enumerate(dev):
            target = d["suffix_ids"][:, -SCORED:]

            logits = receiver_logits(receiver, wte, d["suffix_ids"], None)
            nll, top1 = per_doc_nll_top1(logits, target)
            no_prefix_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

            zero_latent = torch.zeros(NUM_LATENT_TOKENS, 768)
            logits = receiver_logits(receiver, wte, d["suffix_ids"], zero_latent)
            nll, top1 = per_doc_nll_top1(logits, target)
            zero_sender_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

            z = bridge(d["H"])
            logits = receiver_logits(receiver, wte, d["suffix_ids"], z)
            nll, top1 = per_doc_nll_top1(logits, target)
            chosen_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

            other_H = dev[(i - 1) % n]["H"]
            z_shuf = bridge(other_H)
            logits = receiver_logits(receiver, wte, d["suffix_ids"], z_shuf)
            nll, top1 = per_doc_nll_top1(logits, target)
            shuffled_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

    ceiling = json.loads(args.ceiling.read_text())
    full_rows = ceiling["per_document"]["receiver_full_context"]
    assert [r["sha256"] for r in no_prefix_rows] == [r["sha256"] for r in full_rows], \
        "dev document set/order must match the receiver-ceiling artifact exactly"

    gains = [a["nll"] - c["nll"] for a, c in zip(no_prefix_rows, chosen_rows)]
    content_advantage = [s["nll"] - c["nll"] for s, c in zip(shuffled_rows, chosen_rows)]
    zero_advantage = [z["nll"] - c["nll"] for z, c in zip(zero_sender_rows, chosen_rows)]
    headroom = [a["nll"] - f["nll"] for a, f in zip(no_prefix_rows, full_rows)]
    gain_mean = sum(gains) / len(gains)
    headroom_mean = sum(headroom) / len(headroom)
    fraction = gain_mean / headroom_mean if headroom_mean > 0 else float("nan")

    finite = all(
        math.isfinite(r["nll"]) and math.isfinite(r["top1"])
        for rows in (no_prefix_rows, zero_sender_rows, chosen_rows, shuffled_rows) for r in rows
    )
    gain_ci = bootstrap_ci(gains, SEED)
    content_ci = bootstrap_ci(content_advantage, SEED + 1)
    zero_ci = bootstrap_ci(zero_advantage, SEED + 2)
    signal = finite and content_ci[0] > 0

    result = {
        "seed": SEED, "corpus_sha256": corpus_sha256, "test_tokenized_or_scored": False,
        "train_documents": TRAIN_DOCS, "dev_documents": DEV_DOCS,
        "sender_layer": SENDER_LAYER, "num_latent_tokens": NUM_LATENT_TOKENS,
        "bridge": "linear_pooling_prefix_tokens_norm_calibrated_prepended_to_input_embeds",
        "target_norm_from_receiver_embedding_median": target_norm,
        "trainable_parameter_count": bridge.param_count(),
        "bridge_sha256": bridge_sha256,
        "epochs_run": len(train_loss_history), "wall_seconds": wall_s,
        "best_dev_loss_at_selection": best_dev,
        "train_loss_history": train_loss_history,
        "dev_eval_epochs": dev_eval_epochs, "dev_loss_history": dev_history,
        "still_improving_at_cutoff": still_improving,
        "dev": {
            "no_prefix": {"nll": sum(r["nll"] for r in no_prefix_rows) / n, "top1": sum(r["top1"] for r in no_prefix_rows) / n},
            "zero_sender": {"nll": sum(r["nll"] for r in zero_sender_rows) / n, "top1": sum(r["top1"] for r in zero_sender_rows) / n},
            "correct": {"nll": sum(r["nll"] for r in chosen_rows) / n, "top1": sum(r["top1"] for r in chosen_rows) / n},
            "shuffled_document": {"nll": sum(r["nll"] for r in shuffled_rows) / n, "top1": sum(r["top1"] for r in shuffled_rows) / n},
            "receiver_matched_headroom_nll": headroom_mean,
            "fraction_of_receiver_matched_headroom_closed": fraction,
            "no_prefix_minus_correct_nll": gain_mean,
            "no_prefix_minus_correct_bootstrap_95pct_ci": gain_ci,
            "shuffled_minus_correct_nll": sum(content_advantage) / len(content_advantage),
            "shuffled_minus_correct_bootstrap_95pct_ci": content_ci,
            "zero_sender_minus_correct_nll": sum(zero_advantage) / len(zero_advantage),
            "zero_sender_minus_correct_bootstrap_95pct_ci": zero_ci,
        },
        "finite": finite,
        "verdict": (
            "LATENT_PREFIX_NORM_CALIBRATED_TRAINING_INCONCLUSIVE" if still_improving else
            "LATENT_PREFIX_NORM_CALIBRATED_DOCUMENT_SPECIFIC_TRANSFER" if signal else
            "LATENT_PREFIX_NORM_CALIBRATED_NO_DOCUMENT_SPECIFIC_TRANSFER"
        ),
        "per_document_dev": {
            "no_prefix": no_prefix_rows, "zero_sender": zero_sender_rows,
            "correct": chosen_rows, "shuffled_document": shuffled_rows,
        },
    }
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded)
    print(f"gain={gain_mean:.6f} gain_ci={gain_ci} shuffled_minus_correct={result['dev']['shuffled_minus_correct_nll']:.6f} "
          f"content_ci={content_ci} zero_ci={zero_ci} fraction={fraction:.6f} verdict={result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
