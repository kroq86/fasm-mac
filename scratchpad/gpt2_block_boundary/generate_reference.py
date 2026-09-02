"""External test-oracle generator for the GPT-2 block-0 differential gate.

Isolated, disposable lab-measurement script -- NOT part of fasm-mac's
production build or runtime. The virtual environment and output directory
must be outside the repository; scripts/prepare-gpt2-block0-oracle.sh provides
the pinned recipe.

Produces ONLY reference/oracle tensors (fixed input hidden state + named
intermediate boundaries + final block output) plus a small provenance
sidecar. It does NOT dump the model's weights -- the production C gate
(fasm/spikes/tensor_gpt2_block0_differential_check.c) reads block 0's
weights directly from the real model.safetensors file via this project's
own self-written safetensors parser, never from a hand-copied .f32 file.

Model identity: huggingface.co/gpt2 (the real 12-layer/124M GPT-2 small
checkpoint, NOT the 6-layer DistilGPT2 that happened to be present locally
at /Users/ll/distilgpt2 -- checked and rejected for this purpose), revision
and artifact SHA-256 recorded into gate_config.txt below and verified at
gate runtime by the C check's own SHA-256 implementation.
"""
import torch, hashlib, json, os, sys
import transformers, huggingface_hub, safetensors, numpy
from transformers import GPT2Model

REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
EXPECTED_WEIGHT_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
OUT = os.environ.get("GPT2_ORACLE_OUT")
if not OUT:
    raise SystemExit("GPT2_ORACLE_OUT must name an output directory outside the repository")
OUT = os.path.realpath(OUT)
REPO = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.commonpath([OUT, REPO]) == REPO:
    raise SystemExit("refusing to create oracle output inside the repository")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
model = GPT2Model.from_pretrained("gpt2", revision=REVISION)
model.eval()
cfg = model.config
assert cfg.n_layer == 12 and cfg.n_embd == 768 and cfg.n_head == 12 and cfg.vocab_size == 50257, \
    "not the real 124M gpt2 config -- refusing to generate a reference fixture against the wrong model"

from huggingface_hub import scan_cache_dir
fingerprints = {}
for repo in scan_cache_dir().repos:
    if repo.repo_id == "gpt2":
        for rev in repo.revisions:
            if rev.commit_hash != REVISION:
                continue
            for f in rev.files:
                if f.file_name == "model.safetensors":
                    h = hashlib.sha256()
                    with open(f.file_path, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1 << 20), b""):
                            h.update(chunk)
                    fingerprints[f.file_name] = h.hexdigest()

if fingerprints.get("model.safetensors") != EXPECTED_WEIGHT_SHA256:
    raise SystemExit("pinned model.safetensors missing or fingerprint mismatch")

# fixed, arbitrary token ids -- no tokenizer involved anywhere in this gate
token_ids = [464, 3290, 3332, 2159, 0, 1, 50256, 464]
T = len(token_ids)
tok = torch.tensor([token_ids], dtype=torch.long)
pos = torch.arange(T, dtype=torch.long).unsqueeze(0)

b = model.h[0]
with torch.no_grad():
    x0 = model.drop(model.wte(tok) + model.wpe(pos))

    # step through the real block manually, calling its own real submodules,
    # to capture every named intermediate boundary at bit-faithful precision
    ln1_out = b.ln_1(x0)                                    # real post-affine LayerNorm
    qkv_boundary = b.attn.c_attn(ln1_out)                   # [T, 3*M], pre head-split
    attn_output_boundary = b.attn(ln1_out)[0]               # post c_proj, pre-residual
    res1 = x0 + attn_output_boundary
    ln2_out = b.ln_2(res1)                                  # real post-affine LayerNorm
    fc_out = b.mlp.c_fc(ln2_out)
    gelu_boundary = b.mlp.act(fc_out)                       # post GELU (gelu_new)
    mlp_out = b.mlp.c_proj(gelu_boundary)
    res2 = res1 + mlp_out

    # cross-check the manual step-through against calling the block whole
    h1_direct = b(x0)[0]
    assert torch.allclose(h1_direct, res2, atol=1e-6), (h1_direct - res2).abs().max()

def dump(name, t):
    arr = t.detach().numpy().astype("float32")
    arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr
    arr.tofile(f"{OUT}/{name}.f32")
    return list(arr.shape)

manifest = {"model": "gpt2", "revision": REVISION,
            "token_ids": token_ids, "T": T, "n_embd": cfg.n_embd, "n_head": cfg.n_head,
            "layer_norm_epsilon": cfg.layer_norm_epsilon, "weight_fingerprints_sha256": fingerprints,
            "oracle_versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                "transformers": transformers.__version__, "huggingface_hub": huggingface_hub.__version__,
                "safetensors": safetensors.__version__, "numpy": numpy.__version__},
            "note": "reference/oracle tensors only -- weights are read from the real checkpoint by the C loader, not dumped here",
            "shapes": {}, "reference_fingerprints_sha256": {}}

for name, t in [
    ("x0_input_hidden", x0), ("h1_reference_output", res2),
    ("boundary_ln1_real_affine", ln1_out), ("boundary_qkv", qkv_boundary),
    ("boundary_attn_output", attn_output_boundary), ("boundary_res1", res1),
    ("boundary_ln2_real_affine", ln2_out), ("boundary_gelu", gelu_boundary),
]:
    manifest["shapes"][name] = dump(name, t)
    with open(f"{OUT}/{name}.f32", "rb") as fh:
        manifest["reference_fingerprints_sha256"][f"{name}.f32"] = hashlib.sha256(fh.read()).hexdigest()

with open(f"{OUT}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

with open(f"{OUT}/reference_block0_sha256.txt", "w") as f:
    for name, digest in manifest["reference_fingerprints_sha256"].items():
        f.write(f"{digest}  {name}\n")

weight_file = "model.safetensors"
with open(f"{OUT}/gate_config.txt", "w") as f:
    f.write(f"T={T}\nM={cfg.n_embd}\nH={cfg.n_head}\nD={cfg.n_embd // cfg.n_head}\nQW={3*cfg.n_embd}\nF={4*cfg.n_embd}\n")
    f.write(f"layer_norm_epsilon={cfg.layer_norm_epsilon}\n")
    f.write(f"weight_file={weight_file}\n")
    f.write(f"weight_sha256={fingerprints[weight_file]}\n")

print("wrote manifest.json + gate_config.txt + reference .f32 tensors to", OUT)
print("h1 stats: mean=%.6f std=%.6f min=%.6f max=%.6f" % (res2.mean().item(), res2.std().item(), res2.min().item(), res2.max().item()))
