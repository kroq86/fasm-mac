"""Trained-translator rerun of the full-bandwidth passthrough
(phase1_full_passthrough_no_compression.py), isolating "was there no
adapter/training" from "was there not enough bandwidth".

The untrained full-passthrough run (all 64 withheld-prefix positions'
norm-calibrated sender states, zero trainable parameters) was
significantly worse than even zero_sender (zero_sender_minus_correct CI
[-0.386, -0.214], i.e. 64 real untranslated vectors hurt more than 64
zero vectors) and no better than shuffled on the primary gate (CI
[-0.0528, 0.0316]). That conflated two different questions: 64 raw,
untranslated sender-space vectors dropped into receiver-space with no
adaptation at all is a different (and harder) condition than the
8-latent-token rung, which at least had a trained projector. This run
adds exactly the missing piece: a single shared linear "translator"
`W` (768x768, no bias) applied independently to each of the 64 prefix
positions (not compressing the sequence -- no cross-position pooling,
unlike the 8-token rung; the only new degree of freedom is per-channel
translation, applied position-wise), followed by the same norm
calibration to the receiver's real median embedding norm used throughout.
Objective is plain NLL, not contrastive -- holding training objective
fixed at what already worked mechanically, changing only bandwidth
(8 pooled tokens -> 64 unpooled, individually-translated tokens) relative
to the norm-calibrated 8-token rung.

W=0 at init is safe: single multiplicative layer per position (same
argument as the dense-no-bias rung), non-zero gradient generically as
long as sender states are non-zero. Unlike dense-no-bias (which saw only
64 single-vector training examples and produced an invalid,
severely-overfit test), this shared W is applied to all 64 positions of
all 64 documents per epoch -- effectively 4096 (position, document)
instances per epoch for the same 768x768=589,824-parameter matrix, a much
better-determined regime than dense-no-bias's 64 examples.

Same four arms, same gate: primary correct-vs-shuffled paired
document-bootstrap CI > 0; secondary correct-vs-no_prefix,
correct-vs-zero_sender. If this also fails to separate correct from
shuffled, bandwidth-with-training is closed as an explanation too, per
the user's own stated stop condition.
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
NUM_LATENT_TOKENS = PREFIX  # no compression: one latent token per prefix position
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


class PositionwiseTranslator(nn.Module):
    """latent_raw_p = W @ H_p for each of the PREFIX positions p
    independently (shared W across positions -- no cross-position mixing,
    no sequence compression). W: (width, width)=(768,768), no bias. Then
    each of the PREFIX output vectors is individually rescaled (no learned
    params in the rescale) to the fixed target_norm (median L2 norm of the
    receiver's real token embeddings), computed once before training.

    W=0 at init is safe: single multiplicative layer, d(loss)/d(W) is
    generically non-zero at W=0 as long as H != 0 (same argument as the
    dense-no-bias rung). With H=0 (zero sender), latent_raw=0 for any W,
    and the rescale of an exact zero vector stays zero -- the zero-sender
    invariant is preserved regardless of W's value."""

    def __init__(self, width: int, target_norm: float):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(width, width))
        self.target_norm = target_norm

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (1, PREFIX, width) -> (PREFIX, width)
        raw = H.squeeze(0) @ self.W.T
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return raw / norm * self.target_norm

    def param_count(self) -> int:
        return self.W.numel()


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

    bridge = PositionwiseTranslator(768, target_norm=target_norm)
    opt = torch.optim.Adam(bridge.parameters(), lr=LR)

    z0 = bridge(train[0]["H"])
    logits0 = receiver_logits(receiver, wte, train[0]["suffix_ids"], z0)
    target0 = train[0]["suffix_ids"][:, -SCORED:]
    loss0 = scored_ce(logits0, target0)
    loss0.backward()
    g = bridge.W.grad
    if g is None or not torch.isfinite(g).all() or g.abs().max().item() == 0.0:
        raise RuntimeError("positionwise-translator initialization has no learning signal")
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
        "bridge": "positionwise_translator_full_bandwidth_norm_calibrated_prepended_to_input_embeds",
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
            "FULL_BANDWIDTH_TRANSLATOR_TRAINING_INCONCLUSIVE" if still_improving else
            "FULL_BANDWIDTH_TRANSLATOR_DOCUMENT_SPECIFIC_TRANSFER" if signal else
            "FULL_BANDWIDTH_TRANSLATOR_NO_DOCUMENT_SPECIFIC_TRANSFER"
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
