# Frozen ordering bridge: native end-to-end scenario

## What runs

External 32-symbol A/B string -> own BPE -> own GPT-2 124M sender -> selected
K/V -> frozen rank-8 residual adapter -> own six-layer DistilGPT2 receiver ->
one class token. Source prefix is `Sequence:` followed by space-separated
symbols; receiver query is `Class:`. These are the original trained task's
formats, not a general question-answering interface. The actual output is the
model's argmax over all 50,257 tokens, not a hardcoded majority classifier.

`tensor-kv-handoff` is a separate bounded executable using the project's shared
GPT-2 primitives and FASM canonical cached-attention executor. It is not yet a
`tensorctl` subcommand. It does NOT call PyTorch or Transformers. Python's
standard library launcher validates bundle hashes before invoking it.
Both models and the 147,456-parameter adapter are frozen; no training is run.

## Run on this machine

```sh
bash scripts/build-kv-handoff.sh
python3 scripts/kv_handoff.py \
  --assets /Users/ll/.cache/fasm-mac-oracles/kv-handoff-ordering-v1 \
  --sequence AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB
```

The result contains predicted token, decoded answer, prefix length and final
receiver cache length. `target_token=-1` means an interactive input has no
supplied ground-truth label. Try another 32-symbol string, but predictions on
inputs outside the trained sequence distribution are not validated guarantees.

## Reproduce from a checkout

Requirements: macOS x86_64 or Apple Silicon with Rosetta, FASM, Xcode command-line
tools/clang, Accelerate, and Python 3 for hash validation. Preparation additionally
needs an **external** torch/transformers/safetensors environment and the pinned
original model files. No automatic package installation, download or training.

```sh
# Choose an external environment. The following is the one used locally.
/Users/ll/.cache/fasm-mac-oracles/gpt2-block0-venv311/bin/python \
  scripts/prepare_kv_handoff.py \
  --sender /absolute/path/to/gpt2-snapshot \
  --receiver /absolute/path/to/distilgpt2 \
  --out /absolute/path/to/NEW-bundle

bash scripts/check_kv_handoff.sh /absolute/path/to/NEW-bundle
```

Sender input is the original `model.safetensors`; receiver input is the original
`pytorch_model.bin`, both SHA-pinned in the preparer. The existing bridge `.pt`
is also SHA-pinned and loaded weights-only. One-time preparation exports the
receiver's named transformer parameters to safetensors losslessly (no reshapes,
weight tuning or handwritten `.f32` model conversion), plus bridge safetensors.
Native code reads these with the existing self-written safetensors loader.
The two base checkpoints are not the same file/model and have separate hashes.

Bundle manifest includes file hashes, oracle versions, layer map, dtype/layout,
rank and numerical tolerance. Keep the whole bundle together. It is approximately
0.9 GiB including both models and 85 MiB of reference tensors, deliberately outside
Git. Preparation refuses an existing output directory and publishes its staging
directory only after successful export. Missing artifacts cause error, never SKIP.
The manifest is a trusted artifact descriptor, not a signed security attestation.

## Acceptance and scope

The gate rebuilds from source, then compares all 32 **original dev** examples:

- selected sender K/V for each of six layers;
- all adapted K/V, before the receiver consumes its query;
- all 50,257 final logits and their argmax.

Elementwise criterion fixed before running: `abs(native-reference) <= 1e-3 +
1e-4*abs(reference)` with explicit non-finite rejection. The oracle uses frozen
PyTorch CPU/eager attention, one document per forward. Native uses x86_64
Accelerate for projections and canonical cached attention token by token.
This is not a bitwise or performance-equivalence requirement. The first complete
native differential reproduced all 32 reference answers (31 correct labels,
including the same error at case 30); worst absolute tensor delta was
0.00103759766, passing the combined absolute/relative bound without relaxation.

The gate also tests repeated invocation output replay and malformed inputs/bundle
metadata. A native contract test covers invalid layer counts/token ids, empty/full
or inconsistent cache positions, NaN inputs, occupied receiver import, overflow
rollback and independent source/receiver storage. There is no claim that this
small test suite exhausts all malformed safetensors or verifies arbitrary models.

Final full gate exited 0; its output is saved in `native_gate.log` alongside
this document. Contract tests, all 32 differentials, byte-identical invocation
replay and invalid bundle/input rejection passed. The separately rebuilt default
binary also returned token 657 (` 0`) for the example command above.

Cache ownership: the sender remains intact; the adapter writes receiver-owned
buffers. Position is committed only after validating the complete adapted cache.
Receiver starts empty, imported prefix length is the starting query position,
and each receiver token extends the canonical Tensor state once. Re-import into
an occupied receiver is rejected. This is not arbitrary cache concatenation,
rebasing, multi-sender merging, or chat-memory transfer.

Existing GPT-2 generation code is unchanged. Its shared loader was extended with
an explicit layer count and the original twelve-layer wrapper retained. No
assembly kernel, attention semantics, checkpoint or tolerance was modified.
Old CLI and required cached-generation gates were rerun after that addition.

Verdict: implementation/integration proof for this checkpoint and geometry.
Not new independent statistical evidence, not the separate SQL-lookup adapter,
not proof of generalization, speedup, compression, or scientific novelty.
