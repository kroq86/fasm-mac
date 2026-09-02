"""External test-oracle generator: real GPT-2 124M greedy generation, token
by token, for several fixed-token-id prompts. Same isolated/disposable-venv
convention as generate_reference.py -- NOT part of fasm-mac's production
build. Run with a fresh external venv (python3.11 + torch==2.13.0 +
transformers==5.16.1, matching scripts/prepare-gpt2-block0-oracle.sh):

    python3.11 -m venv /path/outside/repo/venv
    /path/outside/repo/venv/bin/pip install torch==2.13.0 transformers==5.16.1
    /path/outside/repo/venv/bin/python3 \
        scratchpad/gpt2_block_boundary/generate_greedy_reference.py

Produces greedy_reference.json/.txt: every generated token for 4 prompts,
for this project's own C generation gate
(fasm/spikes/tensor_gpt2_generation_differential_check.c) to compare
token-for-token, not just a single anchored logits snapshot.

Known environment quirk, not a bug in this project: a length-1 initial
sequence reliably crashes this torch/transformers build on this machine
with SIGBUS inside the C++ extension (reproduced directly, no Python
traceback) -- so no length-1 prompt is included here. The shortest prompt
tested is 2 tokens, which works fine.
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

prompts = [
    [464, 3290, 3332, 2159],   # matches the existing block/full-model anchor's fixed tokens
    [15496, 11, 995, 0],       # "Hello, world!" (real BPE ids for this exact text)
    [1, 2, 3, 4, 5],           # small/low ids, out-of-distribution -- greedy degenerates into a
                               # repeating loop (2,5,2,5,...), a real and useful edge case: this
                               # project's generation loop must reproduce that exact repetition,
                               # not just plausible-looking text
    [464, 3290],               # short 2-token prompt
]
gen_len = 12

results = []
with torch.no_grad():
    for prompt in prompts:
        ids = list(prompt)
        for _ in range(gen_len):
            tok = torch.tensor([ids], dtype=torch.long)
            logits = model(tok).logits
            next_id = int(logits[0, -1].argmax().item())
            ids.append(next_id)
        results.append({"prompt": prompt, "generated": ids[len(prompt):], "full_sequence": ids})
        print(prompt, "->", ids)

with open(f"{OUT}/greedy_reference.json", "w") as f:
    json.dump({"gen_len": gen_len, "cases": results}, f, indent=2)

with open(f"{OUT}/greedy_reference.txt", "w") as f:
    for r in results:
        p = r["prompt"]; g = r["generated"]
        f.write(f"{len(p)} {' '.join(map(str, p))} {len(g)} {' '.join(map(str, g))}\n")

print("wrote greedy_reference.json + greedy_reference.txt to", OUT)
