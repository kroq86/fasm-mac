#!/usr/bin/env python3
"""Train/evaluate the frozen per-channel latent-alignment rung on train/dev."""
from __future__ import annotations

import argparse, hashlib, json, math, random, struct
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from phase1_scalar_sweep import (DEV_DOCS, PREFIX, SCORED, SEED, SUFFIX,
    bootstrap_ci, bucket, load_documents, select, sha)

TRAIN_DOCS, BATCH, MAX_EPOCHS, PATIENCE, LR = 64, 8, 30, 5, 1e-2


class PerChannel(torch.nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(width))
        self.bias = torch.nn.Parameter(torch.zeros(width))

    def forward(self, h):
        return h * self.scale + self.bias


def prepare(sender, tokenizer, records):
    prefixes = torch.tensor([r["ids"][:PREFIX] for r in records])
    suffixes = torch.tensor([r["ids"][PREFIX:PREFIX + SUFFIX] for r in records])
    receiver_ids = torch.cat((torch.full((len(records), 1), tokenizer.eos_token_id), suffixes), 1)
    states = []
    with torch.inference_mode():
        for start in range(0, len(records), BATCH):
            out = sender(prefixes[start:start+BATCH], use_cache=False, output_hidden_states=True)
            h = out.hidden_states[6][:, -1, :].float()
            states.append(torch.nn.functional.layer_norm(h, (h.shape[-1],)))
    return {"hashes": [r["sha256"] for r in records],
            "full_ids": torch.tensor([r["ids"] for r in records]),
            "receiver_ids": receiver_ids, "states": torch.cat(states)}


class Injection:
    def __init__(self, receiver):
        self.receiver, self.extra = receiver, None
        self.handle = receiver.transformer.h[3].register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        if self.extra is None:
            return None
        hidden = args[0].clone()
        hidden[:, 0, :] += self.extra
        return (hidden,) + args[1:]

    def run(self, ids, extra):
        self.extra = extra
        try:
            return self.receiver(ids, use_cache=False).logits
        finally:
            self.extra = None

    def close(self):
        self.handle.remove()


def loss_top1(logits, ids):
    pred, target = logits[:, -SCORED-1:-1].float(), ids[:, -SCORED:]
    losses = torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]),
        target.reshape(-1), reduction="none").reshape(ids.shape[0], SCORED)
    return losses.mean(1), (pred.argmax(-1) == target).float().mean(1)


def evaluate(injector, bridge, data, mode="correct"):
    rows = []
    with torch.inference_mode():
        states = data["states"].roll(1, 0) if mode == "shuffled" else data["states"]
        if mode == "zero":
            states = torch.zeros_like(states)
        for start in range(0, len(data["hashes"]), BATCH):
            ids = data["receiver_ids"][start:start+BATCH]
            extra = None if bridge is None else bridge(states[start:start+BATCH])
            nll, top1 = loss_top1(injector.run(ids, extra), ids)
            for j in range(len(ids)):
                rows.append({"sha256": data["hashes"][start+j],
                    "nll": nll[j].item(), "top1": top1[j].item()})
    return rows


def train(receiver, train_data, dev_data, zero_sender=False):
    bridge, injector = PerChannel(), Injection(receiver)
    optimizer = torch.optim.Adam(bridge.parameters(), lr=LR)
    best, best_state, best_epoch, stale, history = math.inf, None, -1, 0, []
    try:
        for epoch in range(MAX_EPOCHS):
            order = torch.randperm(TRAIN_DOCS, generator=torch.Generator().manual_seed(SEED + epoch))
            for start in range(0, TRAIN_DOCS, BATCH):
                idx, ids = order[start:start+BATCH], train_data["receiver_ids"][order[start:start+BATCH]]
                states = train_data["states"][idx]
                if zero_sender:
                    states = torch.zeros_like(states)
                optimizer.zero_grad()
                nll, _ = loss_top1(injector.run(ids, bridge(states)), ids)
                nll.mean().backward()
                if epoch == 0 and start == 0:
                    for name, p in bridge.named_parameters():
                        if p.grad is None or not torch.isfinite(p.grad).all():
                            raise RuntimeError(f"invalid first gradient: {name}")
                    # With a deliberately zero sender, scale has no learning
                    # signal by construction; this is the control's point.
                    # Bias must still be live. In the real arm both must be.
                    if bridge.bias.grad.abs().max().item() == 0:
                        raise RuntimeError("zero-gradient bias initialization")
                    if not zero_sender and bridge.scale.grad.abs().max().item() == 0:
                        raise RuntimeError("zero-gradient scale initialization")
                optimizer.step()
            rows = evaluate(injector, bridge, dev_data, "zero" if zero_sender else "correct")
            value = sum(r["nll"] for r in rows) / len(rows)
            history.append(value)
            if value < best - 1e-5:
                best, best_epoch, stale = value, epoch, 0
                best_state = {k: v.detach().clone() for k, v in bridge.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    finally:
        injector.close()
    bridge.load_state_dict(best_state)
    return bridge.eval(), {"best_epoch": best_epoch, "epochs_run": len(history), "nll_history": history}


def summary(rows):
    return {"nll": sum(r["nll"] for r in rows)/len(rows),
            "top1": sum(r["top1"] for r in rows)/len(rows)}


def parameter_hash(*bridges):
    payload = bytearray()
    for bridge in bridges:
        for name, tensor in sorted(bridge.state_dict().items()):
            encoded = name.encode()
            payload += struct.pack("<I", len(encoded)) + encoded
            payload += tensor.detach().numpy().astype("<f4").tobytes()
    return hashlib.sha256(payload).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True); ap.add_argument("--receiver", required=True)
    ap.add_argument("--corpus", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(SEED); torch.use_deterministic_algorithms(True); random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(args.sender, local_files_only=True).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver, local_files_only=True).eval()
    for model in (sender, receiver):
        for p in model.parameters(): p.requires_grad_(False)
    docs = load_documents(args.corpus)
    split_hashes = {split: sha(("\n".join(sorted(sha(d.encode()) for d in docs if bucket(d)==split))+"\n").encode()) for split in ("train","dev","test")}
    train_data = prepare(sender, tokenizer, select(tokenizer, docs, "train", TRAIN_DOCS))
    dev_data = prepare(sender, tokenizer, select(tokenizer, docs, "dev", DEV_DOCS))
    bridge, train_meta = train(receiver, train_data, dev_data)
    zero_bridge, zero_meta = train(receiver, train_data, dev_data, True)
    injector = Injection(receiver)
    neutral = evaluate(injector, None, dev_data); correct = evaluate(injector, bridge, dev_data)
    shuffled = evaluate(injector, bridge, dev_data, "shuffled")
    zero_control = evaluate(injector, zero_bridge, dev_data, "zero"); injector.close()
    full = []
    with torch.inference_mode():
        for start in range(0, DEV_DOCS, BATCH):
            ids = dev_data["full_ids"][start:start+BATCH]
            nll, top1 = loss_top1(receiver(ids, use_cache=False).logits, ids)
            for j in range(len(ids)): full.append({"sha256": dev_data["hashes"][start+j], "nll": nll[j].item(), "top1": top1[j].item()})
    comparisons = {}
    for label, rows in (("neutral",neutral),("shuffled",shuffled),("zero_sender",zero_control)):
        values = [a["nll"]-b["nll"] for a,b in zip(rows,correct)]
        comparisons[label+"_minus_correct"] = {"mean": sum(values)/len(values), "bootstrap_95pct_ci": bootstrap_ci(values, SEED+len(comparisons))}
    headroom = sum(a["nll"]-b["nll"] for a,b in zip(neutral,full))/DEV_DOCS
    fraction = comparisons["neutral_minus_correct"]["mean"]/headroom
    finite = all(math.isfinite(r[k]) for rows in (neutral,correct,shuffled,zero_control,full) for r in rows for k in ("nll","top1"))
    sufficient = finite and fraction >= .20 and all(v["bootstrap_95pct_ci"][0] > 0 for v in comparisons.values())
    result = {"seed":SEED,"corpus_sha256":sha(args.corpus.read_bytes()),"split_document_hashes_sha256":split_hashes,
        "test_tokenized_or_scored":False,"protocol":{"train_documents":TRAIN_DOCS,"dev_documents":DEV_DOCS,"batch":BATCH,"max_epochs":MAX_EPOCHS,"patience":PATIENCE,"learning_rate":LR},
        "bridge":"per_channel_affine_additive","trainable_parameter_count":1536,"parameter_sha256":parameter_hash(bridge,zero_bridge),
        "training":train_meta,"zero_sender_training":zero_meta,
        "dev":{"neutral":summary(neutral),"correct":summary(correct),"shuffled_document":summary(shuffled),"zero_sender":summary(zero_control),"full_context_receiver":summary(full),"headroom_nll":headroom,"fraction_of_headroom_closed":fraction,"comparisons":comparisons},
        "finite":finite,"verdict":"PER_CHANNEL_SUFFICIENT" if sufficient else "PER_CHANNEL_INSUFFICIENT",
        "per_document_dev":{"neutral":neutral,"correct":correct,"shuffled_document":shuffled,"zero_sender":zero_control,"full_context_receiver":full}}
    encoded=json.dumps(result,sort_keys=True,indent=2)+"\n"; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(encoded); print(encoded,end="")


if __name__ == "__main__": main()
