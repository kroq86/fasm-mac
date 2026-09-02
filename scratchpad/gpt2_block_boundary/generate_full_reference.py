"""External test-oracle generator for the GPT-2 full-model (12 layers + ln_f
+ tied LM head) differential gate. Same isolated/disposable-venv convention
as generate_reference.py -- NOT part of fasm-mac's production build.

    scratchpad/gpt2_block_boundary/venv311/bin/python3 \
        scratchpad/gpt2_block_boundary/generate_full_reference.py

Produces ONLY reference/oracle tensors (two mid-stack layer outputs, the
post-ln_f hidden state, and the final tied-head logits) -- weights are read
directly from the real model.safetensors by the production C loader, never
dumped here.

Note on this transformers version (5.16.1): GPT2Block.forward returns a bare
Tensor, not a (tensor, ...) tuple as in older releases -- `x = block(x)`
directly, NOT `block(x)[0]` when chaining multiple blocks (indexing [0] on
a bare [1,T,M] tensor silently squeezes the batch dim instead of unpacking a
tuple, which only stays harmless for a single non-chained call).
"""
import torch, json, os
from transformers import GPT2LMHeadModel

torch.manual_seed(0)
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
cfg = model.config
assert cfg.n_layer == 12 and cfg.n_embd == 768 and cfg.n_head == 12 and cfg.vocab_size == 50257, \
    "not the real 124M gpt2 config -- refusing to generate a reference fixture against the wrong model"

OUT = os.path.dirname(os.path.abspath(__file__))

token_ids = [464, 3290, 3332, 2159, 0, 1, 50256, 464]
T = len(token_ids)
tok = torch.tensor([token_ids], dtype=torch.long)
pos = torch.arange(T, dtype=torch.long).unsqueeze(0)

tr = model.transformer
with torch.no_grad():
    x = tr.drop(tr.wte(tok) + tr.wpe(pos))
    layer_outputs = []
    for block in tr.h:
        x = block(x)
        layer_outputs.append(x.clone())
    final_hidden = tr.ln_f(x)
    logits = model.lm_head(final_hidden)

    out = model(tok)
    assert torch.allclose(out.logits, logits, atol=1e-6), (out.logits - logits).abs().max()

def dump(name, t):
    arr = t.detach().numpy().astype("float32")
    arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr
    arr.tofile(f"{OUT}/{name}.f32")
    return list(arr.shape)

manifest = {"token_ids": token_ids, "T": T, "n_embd": cfg.n_embd, "n_head": cfg.n_head, "n_layer": cfg.n_layer,
            "vocab_size": cfg.vocab_size, "layer_norm_epsilon": cfg.layer_norm_epsilon,
            "note": "full-model reference/oracle tensors only", "shapes": {}}

manifest["shapes"]["full_layer5_output"] = dump("full_layer5_output", layer_outputs[5])
manifest["shapes"]["full_layer11_output"] = dump("full_layer11_output", layer_outputs[11])
manifest["shapes"]["full_ln_f_output"] = dump("full_ln_f_output", final_hidden)
manifest["shapes"]["full_logits"] = dump("full_logits", logits)

with open(f"{OUT}/manifest_full.json", "w") as f:
    json.dump(manifest, f, indent=2)

top5 = logits[0, -1].topk(5)
print("last-position top5 token ids:", top5.indices.tolist(), "logits:", [round(v, 4) for v in top5.values.tolist()])
print("logits stats: mean=%.6f std=%.6f min=%.6f max=%.6f" % (logits.mean().item(), logits.std().item(), logits.min().item(), logits.max().item()))
print("wrote full-model reference tensors to", OUT)
