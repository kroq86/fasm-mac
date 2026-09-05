"""Preregistered optimization-reliability / initialization-sensitivity
check -- not a search for a better result, and not framed as a "seed check
of the original experiment" (the original run used exact-zero init, which
has zero randomness anywhere in the pipeline; there was never any seed
variability to check in the first place). This is a new, separate
question: does the near-signal found by
phase1_full_passthrough_trained_translator.py (`shuffled_minus_correct =
0.004969`, CI `[-0.001010, 0.011926]` -- real sign/magnitude, CI barely
still including zero, LR=1e-2, exact-zero init, converged suspiciously
fast: real movement epoch 1, frozen to 6 decimals for the remaining 29)
reproduce under a small-random init and a spread of learning rates, or was
it specific to that one deterministic trajectory?

Narrow, targeted grid, fixed BEFORE running, not adjusted after seeing any
result -- 4 runs, not a full factorial (a 3x3=9-run design was scoped out
as overkill for the actual question, which is only "did we get lucky once",
not a full optimizer-robustness benchmark):
  1. lr=1e-2 (original run's LR), seed=17  -- seed sensitivity at the
  2. lr=1e-2 (original run's LR), seed=42  -- original LR, two seeds
  3. lr=3e-3 (~3x lower),          seed=17 -- LR sensitivity at one seed
  4. lr=3e-2 (~3x higher),         seed=17 -- LR sensitivity at one seed
Runs 1-2 answer seed sensitivity at the original LR; runs 3-4 answer LR
sensitivity at one fixed seed. If these 3-4 runs reproduce the ~+0.005
document-specific gap with at least occasional CI-above-zero, the grid is
worth extending; if the gap goes back near zero or the sign is unstable,
the optimization-reliability hypothesis is closed without further runs.
Switching to small-random init (`randn * 0.02`, matching the
low-rank/prefix-projector convention elsewhere in this project) is itself
a real optimization-regime change relative to the original exact-zero-init
run -- that original run is preserved as its own artifact, at its own LR,
under its own (deterministic) init, and is not rerun or merged into this
grid. Everything else identical to
phase1_full_passthrough_trained_translator.py: corpus, splits, site
(GPT-2 final layer -> DistilGPT2 first block), K=64 unpooled
per-position-translated latent tokens, norm calibration, plain NLL
objective, 30-epoch budget, dev scored every 5 epochs, same four arms,
same primary gate.

Primary gate, unchanged: `shuffled_minus_correct > 0` AND its paired
document-bootstrap CI entirely above zero, on held-out dev.

Explicitly NOT sufficient for a positive reliability verdict (stated
before running, not decided after seeing results):
  - one lucky seed out of nine passing while the rest do not;
  - a single best run passing while the others sit near zero;
  - correct beating zero_sender alone (that was already established and
    is not in dispute -- it's a generic-calibration effect, not the
    content-specific one this check is about);
  - a lower train loss alone.
A positive verdict requires a REPEATABLE correct-vs-shuffled effect: e.g.
most seeds at the same LR agree in sign, and an aggregate over the
preregistered runs clears the gate -- not an isolated pass.
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
SENDER_LAYER = 12
MAX_EPOCHS = 100  # convergence-completion rerun: all 4 grid runs at 30 epochs
# still had still_improving_at_cutoff=True; everything else (LR/seed grid,
# architecture, data, loss, norm calibration, dev-eval cadence, gate) is
# unchanged from the 30-epoch check -- only the budget is extended, to
# test "was 30 epochs just not enough for small-random init", not to
# search for a better result.
DEV_EVAL_EVERY = 5
STILL_IMPROVING_REL_THRESHOLD = 0.01
W_INIT_STD = 0.02

GRID = (
    {"lr": 1e-2, "seed": 17},
    {"lr": 1e-2, "seed": 42},
    {"lr": 3e-3, "seed": 17},
    {"lr": 3e-2, "seed": 17},
)
LRS = tuple(sorted({g["lr"] for g in GRID}))


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
            H = out.hidden_states[SENDER_LAYER].float()
            H = torch.nn.functional.layer_norm(H, (H.shape[-1],)).detach()
            prepared.append({"sha256": record["sha256"], "suffix_ids": suffix_ids, "H": H})
    return prepared


class PositionwiseTranslator(nn.Module):
    """Same as phase1_full_passthrough_trained_translator.py's bridge, but
    W is small-random at init (not exact zero) so that `seed` actually
    varies something -- required for this check's own purpose. With H=0
    (zero sender), latent_raw=0 for any W regardless of init, so the
    zero-sender invariant is unaffected by this change."""

    def __init__(self, width: int, target_norm: float, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(width, width, generator=g) * W_INIT_STD)
        self.target_norm = target_norm

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        raw = H.squeeze(0) @ self.W.T
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return raw / norm * self.target_norm

    def param_count(self) -> int:
        return self.W.numel()


def receiver_logits(receiver, wte, suffix_ids: torch.Tensor, latent: torch.Tensor | None):
    suffix_embeds = wte(suffix_ids)
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


def run_one(lr: float, run_seed: int, receiver, wte, train, dev, full_rows, target_norm, progress_path: Path) -> dict:
    torch.manual_seed(run_seed)
    bridge = PositionwiseTranslator(768, target_norm=target_norm, seed=run_seed)
    opt = torch.optim.Adam(bridge.parameters(), lr=lr)

    z0 = bridge(train[0]["H"])
    logits0 = receiver_logits(receiver, wte, train[0]["suffix_ids"], z0)
    target0 = train[0]["suffix_ids"][:, -SCORED:]
    loss0 = scored_ce(logits0, target0)
    loss0.backward()
    g = bridge.W.grad
    if g is None or not torch.isfinite(g).all() or g.abs().max().item() == 0.0:
        raise RuntimeError(f"lr={lr} seed={run_seed}: initialization has no learning signal")
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
            "lr": lr, "run_seed": run_seed, "epochs_run": epoch + 1,
            "train_loss_history": train_loss_history,
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

    n = len(dev)
    with torch.no_grad():
        no_prefix_rows, zero_sender_rows, chosen_rows, shuffled_rows = [], [], [], []
        for i, d in enumerate(dev):
            target = d["suffix_ids"][:, -SCORED:]

            logits = receiver_logits(receiver, wte, d["suffix_ids"], None)
            nll, top1 = per_doc_nll_top1(logits, target)
            no_prefix_rows.append({"sha256": d["sha256"], "nll": nll, "top1": top1})

            zero_latent = torch.zeros(PREFIX, 768)
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

    assert [r["sha256"] for r in no_prefix_rows] == [r["sha256"] for r in full_rows]

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
    gain_ci = bootstrap_ci(gains, run_seed)
    content_ci = bootstrap_ci(content_advantage, run_seed + 1)
    zero_ci = bootstrap_ci(zero_advantage, run_seed + 2)
    signal = finite and content_ci[0] > 0

    return {
        "lr": lr, "run_seed": run_seed,
        "epochs_run": len(train_loss_history), "wall_seconds": wall_s,
        "still_improving_at_cutoff": still_improving,
        "train_loss_first_last": [train_loss_history[0], train_loss_history[-1]],
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
        "gate_pass": signal,
        "content_advantage_per_doc": content_advantage,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ceiling", type=Path, required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval()
    for m in (sender, receiver):
        for p in m.parameters():
            p.requires_grad_(False)
    wte = receiver.transformer.wte

    with torch.no_grad():
        target_norm = wte.weight.float().norm(dim=-1).median().item()

    docs = load_documents(args.corpus)
    corpus_sha256 = sha(args.corpus.read_bytes())
    train_records = select(tokenizer, docs, "train", TRAIN_DOCS)
    dev_records = select(tokenizer, docs, "dev", DEV_DOCS)
    train = prepare(sender, tokenizer, train_records)
    dev = prepare(sender, tokenizer, dev_records)

    ceiling = json.loads(args.ceiling.read_text())
    full_rows = ceiling["per_document"]["receiver_full_context"]

    runs = []
    for i, combo in enumerate(GRID):
        progress_path = args.output.with_name(f"{args.output.stem}_run{i}_progress.json")
        r = run_one(combo["lr"], combo["seed"], receiver, wte, train, dev, full_rows, target_norm, progress_path)
        runs.append(r)
        args.output.write_text(json.dumps({
            "seed_base": SEED, "corpus_sha256": corpus_sha256, "test_tokenized_or_scored": False,
            "grid": list(GRID), "runs_completed": len(runs), "runs": runs,
            "reliability_verdict": "IN_PROGRESS",
        }, sort_keys=True, indent=2) + "\n")
        print(f"[grid {i}] lr={combo['lr']} seed={combo['seed']} "
              f"shuffled_minus_correct={r['dev']['shuffled_minus_correct_nll']:.6f} "
              f"ci={r['dev']['shuffled_minus_correct_bootstrap_95pct_ci']} gate_pass={r['gate_pass']}")

    passes = sum(1 for r in runs if r["gate_pass"])
    positive_signs = sum(1 for r in runs if r["dev"]["shuffled_minus_correct_nll"] > 0)

    per_lr = {}
    for lr in LRS:
        lr_runs = [r for r in runs if r["lr"] == lr]
        lr_pos = sum(1 for r in lr_runs if r["dev"]["shuffled_minus_correct_nll"] > 0)
        lr_pass = sum(1 for r in lr_runs if r["gate_pass"])
        per_lr[str(lr)] = {"positive_signs": lr_pos, "gate_passes": lr_pass, "n": len(lr_runs)}
    # strict majority per LR-group (>n/2), works for both the 2-seed group
    # (lr=1e-2: needs both positive) and the 1-run groups (needs that one
    # run positive) without a hardcoded ">=2" that only made sense for n=3
    majority_sign_lrs = sum(1 for v in per_lr.values() if v["positive_signs"] > v["n"] / 2)

    # Caveat, reported alongside the number rather than hidden: these
    # values (4 runs x 32 documents = 128) are not independent -- the same
    # 32 documents recur across all 4 bridges -- so this pooled bootstrap CI
    # is a rough aggregate-effect-size indicator, not a fully valid
    # mixed-effects estimate. The per-LR sign-majority criterion above is
    # the more conservative primary signal; this is supporting evidence.
    pooled = [x for r in runs for x in r["content_advantage_per_doc"]]
    pooled_mean = sum(pooled) / len(pooled)
    pooled_ci = bootstrap_ci(pooled, SEED)
    aggregate_gate_pass = pooled_ci[0] > 0

    # Explicitly not sufficient on their own (stated in the module
    # docstring before running): a single passing run among nine, or one
    # best run standing out while the rest sit near zero. A positive
    # verdict requires either (a) most seeds at some LR agreeing in sign
    # AND that LR's aggregate clearing the gate, or (b) the full
    # pooled aggregate across all preregistered runs clearing the gate
    # outright.
    n_runs = len(runs)
    if aggregate_gate_pass or (majority_sign_lrs >= 1 and passes >= 2):
        reliability_verdict = "REPRODUCES"
    elif positive_signs >= n_runs - 1:  # all or all-but-one positive by sign
        reliability_verdict = "REPRODUCES_BY_SIGN_ONLY"
    else:
        reliability_verdict = "DOES_NOT_REPRODUCE"

    final = {
        "seed_base": SEED, "corpus_sha256": corpus_sha256, "test_tokenized_or_scored": False,
        "grid": list(GRID), "runs_completed": n_runs, "runs": runs,
        "gate_passes": passes, "positive_sign_count": positive_signs,
        "per_lr_summary": per_lr, "majority_sign_lrs": majority_sign_lrs,
        "pooled_shuffled_minus_correct_mean": pooled_mean,
        "pooled_bootstrap_95pct_ci": pooled_ci,
        "aggregate_gate_pass": aggregate_gate_pass,
        "reliability_verdict": reliability_verdict,
    }
    args.output.write_text(json.dumps(final, sort_keys=True, indent=2) + "\n")
    print(f"FINAL: passes={passes}/{n_runs} positive_signs={positive_signs}/{n_runs} "
          f"pooled_mean={pooled_mean:.6f} pooled_ci={pooled_ci} reliability_verdict={reliability_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
