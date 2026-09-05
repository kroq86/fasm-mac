#!/usr/bin/env python3
"""Phase 1: train exactly one small affine bridge, both LMs frozen.

Per scratchpad/gpt2_distilgpt2_latent_bridge_preregistration.md's Phase 1:
z = alpha * (W @ LayerNorm(h_sender) + b). Alpha and bias are zero-initialized
while W starts as identity, so the untrained bridge starts at neutral behavior
without creating a zero-gradient fixed point. Injection is ADDITIVE at
the receiver's prepended slot position (h_slot_natural + z), not
replacive like Phase 0b's mechanism -- only with additive injection does
alpha=0 give literally the neutral_slot baseline, not an arbitrary
all-zero override. W: [768,768] + b: [768] + alpha: [1] = 590,593
trainable params, matching the preregistration's stated count exactly
(LayerNorm has NO learnable affine of its own -- elementwise_affine=False
-- otherwise the count would be off by 2*768).

Scale/scope honesty: this trains on the SAME reduced first-pass corpus as
Phase 0b (80 train / 20 dev synthetic documents), not yet the full
preregistered ~4-5k-document corpus with a held-out test split. No test
split exists yet, so no test-split verdict is computed here -- this is a
train/dev-only first look at whether the bridge can learn anything at
all, gated the same way Phase 0b was: report the real numbers, do not
present a dev-only result as the Phase-1 stop-gate verdict (which
requires an untouched test split per the preregistration).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

SENDER_LAYER = 6
RECEIVER_INJECT_BLOCK = 3
SCORE_LAST_N = 32


class Bridge(nn.Module):
    """Minimum-alignment ladder, per the Mostik-analysis reframing: don't
    start at the most expressive transformation, start at the simplest and
    only add complexity if it fails.

      scalar:   z = alpha * LayerNorm(hS)                          (1 param)
      diagonal: z = alpha * (LayerNorm(hS) * w + b), w,b per-channel (1537 params)
      dense:    z = alpha * (LayerNorm(hS) @ W + b), W full 768x768  (590,593 params)

    All three start at z=0 externally (alpha=0), but W=b=alpha=0
    simultaneously would be a dead parameterization -- every partial
    derivative is zero, so no optimizer can ever move away from it (caught
    independently before this rung was run). Where a W/w exists, it starts
    at identity/ones so alpha has a non-zero first-step gradient; where no
    W exists (scalar), LayerNorm(hS) itself is generically non-zero, so
    alpha's gradient is non-zero from step 0 with no dead point possible.
    """

    def __init__(self, width: int = 768, kind: str = "scalar"):
        super().__init__()
        assert kind in ("scalar", "diagonal", "dense")
        self.kind = kind
        self.width = width
        self.ln = nn.LayerNorm(width, elementwise_affine=False)
        self.alpha = nn.Parameter(torch.zeros(()))
        if kind == "diagonal":
            self.w = nn.Parameter(torch.ones(width))
            self.b = nn.Parameter(torch.zeros(width))
        elif kind == "dense":
            self.W = nn.Parameter(torch.eye(width))
            self.b = nn.Parameter(torch.zeros(width))

    def forward(self, hS: torch.Tensor) -> torch.Tensor:
        normed = self.ln(hS)
        if self.kind == "scalar":
            return self.alpha * normed
        if self.kind == "diagonal":
            return self.alpha * (normed * self.w + self.b)
        return self.alpha * (normed @ self.W.T + self.b)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_injected_forward(receiver, block_index: int):
    """Returns (run(ids, extra_row0), ) where extra_row0 (or None) is
    ADDED to the slot's natural hidden state at block_index's input."""
    state = {"extra": None}
    block = receiver.transformer.h[block_index]

    def hook(_module, args):
        if state["extra"] is None:
            return None
        hidden = args[0].clone()
        hidden[:, 0, :] = hidden[:, 0, :] + state["extra"]
        return (hidden,) + args[1:]

    handle = block.register_forward_pre_hook(hook)

    def run(ids: torch.Tensor, extra_row0: torch.Tensor | None):
        state["extra"] = extra_row0
        try:
            return receiver(input_ids=ids, use_cache=False).logits
        finally:
            state["extra"] = None

    return run, handle


def scored_ce(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    pred = logits[:, :-1]
    target = target_ids[:, 1:]
    losses = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none"
    ).reshape(target.shape)
    n = min(SCORE_LAST_N, losses.shape[1])
    return losses[:, -n:].mean()


def precompute(sender, tokenizer, docs, slot_id):
    out = []
    with torch.no_grad():
        for doc in docs:
            prefix_ids = tokenizer(doc["prefix"], return_tensors="pt")["input_ids"]
            suffix_ids = tokenizer(doc["suffix"], return_tensors="pt")["input_ids"]
            sender_out = sender(input_ids=prefix_ids, output_hidden_states=True, use_cache=False)
            hS = sender_out.hidden_states[SENDER_LAYER][:, -1, :].detach()
            hS_early = sender_out.hidden_states[SENDER_LAYER][:, 0, :].detach()
            slot_and_suffix = torch.cat([torch.tensor([[slot_id]]), suffix_ids], dim=1)
            out.append({"hS": hS, "hS_early": hS_early, "slot_and_suffix": slot_and_suffix})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bridge-out", type=Path, required=True)
    ap.add_argument("--bridge-kind", choices=("scalar", "diagonal", "dense"), default="scalar")
    args = ap.parse_args()

    torch.manual_seed(20260904)
    corpus = json.loads(args.corpus.read_text())
    train_docs, dev_docs = corpus["train"], corpus["dev"]

    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    slot_id = tokenizer.eos_token_id
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval()
    for m in (sender, receiver):
        for p in m.parameters():
            p.requires_grad_(False)

    train_pre = precompute(sender, tokenizer, train_docs, slot_id)
    dev_pre = precompute(sender, tokenizer, dev_docs, slot_id)

    bridge = Bridge(768, kind=args.bridge_kind)
    expected_params = {"scalar": 1, "diagonal": 1 + 768 + 768, "dense": 590_593}
    assert bridge.param_count() == expected_params[args.bridge_kind], bridge.param_count()
    run, handle = make_injected_forward(receiver, RECEIVER_INJECT_BLOCK)
    opt = torch.optim.Adam(bridge.parameters(), lr=args.lr)

    t0 = time.time()
    scored_tokens = 0
    dev_history = []
    best_dev = math.inf
    best_state = {k: v.clone() for k, v in bridge.state_dict().items()}

    def dev_loss() -> float:
        bridge.eval()
        with torch.no_grad():
            total = 0.0
            for d in dev_pre:
                z = bridge(d["hS"])
                logits = run(d["slot_and_suffix"], z)
                total += scored_ce(logits, d["slot_and_suffix"]).item()
        bridge.train()
        return total / len(dev_pre)

    for epoch in range(args.epochs):
        bridge.train()
        epoch_loss = 0.0
        for d in train_pre:
            opt.zero_grad()
            z = bridge(d["hS"])
            logits = run(d["slot_and_suffix"], z)
            loss = scored_ce(logits, d["slot_and_suffix"])
            loss.backward()
            if epoch == 0 and scored_tokens == 0:
                alpha_grad = bridge.alpha.grad
                if alpha_grad is None or not torch.isfinite(alpha_grad) or alpha_grad.abs().item() == 0.0:
                    raise RuntimeError("bridge initialization has no learning signal")
            opt.step()
            epoch_loss += loss.item()
            scored_tokens += SCORE_LAST_N
        dl = dev_loss()
        dev_history.append(dl)
        if dl < best_dev:
            best_dev = dl
            best_state = {k: v.clone() for k, v in bridge.state_dict().items()}
        if time.time() - t0 > 2 * 3600 or scored_tokens > 2_000_000:
            break

    handle.remove()
    bridge.load_state_dict(best_state)
    wall_s = time.time() - t0

    bridge_bytes = sum(p.numel() * 4 for p in bridge.parameters())
    torch.save(bridge.state_dict(), args.bridge_out)
    bridge_sha256 = hashlib.sha256(args.bridge_out.read_bytes()).hexdigest()

    # Final arm comparison on DEV (no test split exists yet for this first pass).
    run2, handle2 = make_injected_forward(receiver, RECEIVER_INJECT_BLOCK)
    torch.manual_seed(20260904)
    random_bridge = Bridge(768, kind=args.bridge_kind)
    with torch.no_grad():
        if args.bridge_kind == "dense":
            random_bridge.W.copy_(torch.empty(768, 768).normal_(0, 0.02))
            random_bridge.b.zero_()
        elif args.bridge_kind == "diagonal":
            random_bridge.w.copy_(torch.empty(768).normal_(1.0, 0.2))
            random_bridge.b.zero_()
        random_bridge.alpha.fill_(1.0)

    def eval_arm(fn) -> tuple[float, float]:
        """fn(i, d) -> injected vector or None, given dev_pre index i and its dict d."""
        losses, tops = [], []
        with torch.no_grad():
            for i, d in enumerate(dev_pre):
                z = fn(i, d)
                logits = run2(d["slot_and_suffix"], z)
                target = d["slot_and_suffix"][:, 1:]
                losses.append(scored_ce(logits, d["slot_and_suffix"]).item())
                pred_logits = logits[:, :-1]
                n = min(SCORE_LAST_N, pred_logits.shape[1])
                top1 = (pred_logits[:, -n:].argmax(-1) == target[:, -n:]).float().mean().item()
                tops.append(top1)
        return sum(losses) / len(losses), sum(tops) / len(tops)

    with torch.no_grad():
        arms = {}
        arms["learned_bridge"] = eval_arm(lambda i, d: bridge(d["hS"]))
        arms["neutral_no_injection"] = eval_arm(lambda i, d: None)
        arms["random_bridge"] = eval_arm(lambda i, d: random_bridge(d["hS"]))
        n_dev = len(dev_pre)
        arms["shuffled_state"] = eval_arm(lambda i, d: bridge(dev_pre[(i + 1) % n_dev]["hS"]))
        arms["wrong_position"] = eval_arm(lambda i, d: bridge(d["hS_early"]))
        arms["receiver_alone_no_slot"] = None  # computed separately below, no slot at all

    handle2.remove()
    with torch.no_grad():
        losses = []
        for doc in dev_docs:
            suffix_ids = tokenizer(doc["suffix"], return_tensors="pt")["input_ids"]
            logits = receiver(input_ids=suffix_ids, use_cache=False).logits
            losses.append(scored_ce(logits, suffix_ids).item())
        arms["receiver_alone_no_slot"] = (sum(losses) / len(losses), None)

    result = {
        "seed": 20260904,
        "scale_note": "first pass, 80 train / 20 dev synthetic docs, no test split yet",
        "sender_layer": SENDER_LAYER,
        "receiver_inject_block": RECEIVER_INJECT_BLOCK,
        "bridge_param_count": bridge.param_count(),
        "bridge_bytes": bridge_bytes,
        "bridge_sha256": bridge_sha256,
        "epochs_run": len(dev_history),
        "scored_tokens_total": scored_tokens,
        "wall_seconds": wall_s,
        "best_dev_loss_nll": best_dev,
        "dev_loss_history": dev_history,
        "final_arms_dev": {k: {"nll": v[0], "top1": v[1]} for k, v in arms.items()},
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    args.output.write_text(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
