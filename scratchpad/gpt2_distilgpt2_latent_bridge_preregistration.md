# Frozen GPT-2 -> DistilGPT2 latent bridge spike

## Claim boundary

Test whether a small learned map can transfer useful next-token information
from a frozen GPT-2 124M hidden state into a frozen DistilGPT2 receiver more
effectively than text/logit handoff controls at lower decode cost than running
the full sender. This is a local model-stitching/latent-communication
experiment, not a claim of novelty or equivalence to Mostik's reported system.

## Verified local facts

- Sender: GPT-2, 12 layers, width 768, 12 heads, vocab 50,257.
- Receiver: DistilGPT2, 6 layers, width 768, 12 heads, vocab 50,257.
- Tokenizer files are byte-identical across both checkpoints:
  `vocab.json` sha256 `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783`,
  `merges.txt` sha256 `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5`.
- Sender is locally available as safetensors; receiver is currently available
  only as a 336 MB `pytorch_model.bin`. Native receiver execution is therefore
  not yet established.

Shared width/tokenizer make the experiment mechanically feasible, but are a
major confound: DistilGPT2 was distilled from GPT-2-family representations, so
success cannot be generalized to unrelated model families.

## Phase 0a — interface sensitivity (no training)

In an external reference environment, capture aligned per-token hidden states
from both frozen models on a fingerprinted calibration corpus. For every
candidate sender/receiver boundary pair, measure:

- identity-map hidden MSE/cosine similarity;
- receiver-suffix next-token NLL/top-1 after identity injection;
- random orthogonal map;
- shuffled-example, shuffled-position, and wrong-layer controls;
- receiver-alone and sender-alone logits.

The instrument passes only if correct aligned identity injection is observably
different from all shuffled/wrong-position controls at the receiver logits. If
the endpoint cannot detect those corruptions, stop before training.

This phase does **not** test information transfer: sender and receiver see the
same token sequence. A lower NLL can arise from representation compatibility or
ordinary activation patching. It authorizes only the next no-training
information/headroom check, not bridge training.

## Phase 0b — withheld-context headroom (no training)

Split each document before tokenization into a 64-token prefix and a 160-token
suffix. The sender sees only the prefix. The receiver sees only a prepended
injection slot plus the suffix. Score only the last 32 suffix targets, so the
injected information must survive a long positional path through the receiver.

Use documents split into train/dev/test before tokenization, with exact and
near-duplicate detection across splits. Phase 0b uses train/dev only. Test
remains unopened until the learned bridge and all choices are frozen.

The minimal-alignment phase is permitted only if all of these hold:

- full-context sender NLL beats receiver suffix-only NLL by at least 0.15
  nats/token, establishing information headroom;
- all arms are finite and two evaluations produce byte-identical evidence.

Identity injection is not required to beat the controls: that is the direct
compatibility hypothesis being tested. If it transfers information, the next
question is whether a learned map is necessary at all. If it does not transfer
despite headroom, the next question is the minimum alignment that changes that
result. Otherwise record `INCONCLUSIVE_NO_HEADROOM` and stop. Do not resize the
context or tune the boundary after observing the failed gate.

## Phase 1 — one trained bridge

Freeze both models. Train exactly one affine bridge from the sender's final
prefix vector into one prepended receiver slot:
`z = alpha * (W * LayerNorm(h_sender) + b)`. Initialize `W` to identity and
`b`, `alpha` to zero so the trained arm starts at receiver-only behavior while
retaining a nonzero gradient path through `alpha`. Initializing all three to
zero is forbidden because it is a zero-gradient fixed point. The bridge has
590,593 trainable parameters; no sender or receiver weight may change. Optimize
receiver next-token cross-entropy only over the frozen 32-token target span.
Boundary/layer selection and stopping use train/dev only. Record bridge
parameter count, training steps and tokens, wall time, peak memory, and artifact
hash. Hard cap: 2 million scored tokens or two hours, whichever comes first.

Dataset documents must be split before tokenization into train/dev/test, with
duplicate and near-duplicate detection. Boundary/layer selection and stopping
use dev only. Test is evaluated once.

## Arms

1. Receiver-alone: suffix only.
2. Sender-alone: suffix only (capability control).
3. Sender-alone: full prefix plus suffix (information ceiling).
4. Literal text handoff using a fixed number of real prefix tokens.
5. Receiver suffix with identity injection.
6. Learned latent bridge.
7. Same learned bridge with examples shuffled.
8. Same learned bridge with token positions shuffled.
9. Random frozen bridge with the same parameter shape.
10. Parameter-count control trained with the sender vector forced to zero.
11. Logit-level teacher handoff/distillation control with comparable learned
    parameter budget, if a matched budget can be defined honestly.

A literal text handoff is reported only if it can be defined without giving an
arm extra generated tokens or compute; otherwise it is marked incomparable,
not forced into the table.

## Primary endpoints

- held-out next-token NLL/perplexity;
- fraction of the receiver-to-sender NLL gap closed;
- top-1 next-token accuracy/agreement;
- total prefill and sustained-decode compute/latency;
- bridge bytes and added activation traffic.

Generation examples are qualitative secondary evidence only.

## Stop gate

`LATENT_TRANSFER_SUPPORTED` requires on the untouched test split:

- learned bridge improves receiver-alone NLL with a paired bootstrap 95% CI
  excluding zero;
- learned bridge beats identity, random, shuffled-state, wrong-position, and
  parameter-count controls under the same scoring protocol;
- at least 20% of the receiver-to-sender NLL gap is closed;
- measured end-to-end compute is lower than full sender inference;
- no non-finite state and deterministic eval replay.

Otherwise record `NO_USEFUL_LATENT_TRANSFER` or `INCONCLUSIVE`. A positive
result is scoped to this related GPT-2 family pair and does not establish
cross-family latent communication.

## Runtime boundary

Phase 0/1 may use PyTorch solely as an external experimental oracle because the
receiver checkpoint is not yet supported by the native loader. Native runtime
work is authorized only after the statistical gate passes. A PyTorch-only pass
is evidence for the mechanism, not evidence that `tensorctl` implements it.

No model download, conversion, runtime modification, large training run,
commit, or push occurs before Phase 0 passes.

Phase 0a and Phase 0b are complete. Phase 0b found substantial information
headroom but raw identity injection made prediction worse. The next active gate
is a sequential minimal-alignment ladder; do not jump directly to an MLP.

### Scalar rung addendum (frozen before its final run)

The first exploratory scalar grid selected its positive boundary (`alpha=2`),
so it did not localize the scalar optimum. One and only one symmetric extension
is permitted before advancing:
`alpha in {0, +/-0.125, +/-0.25, +/-0.5, +/-1, +/-2, +/-4, +/-8}`.
Selection uses the existing 64-document train bucket; the same 32-document dev
bucket remains the architecture-selection surface; test remains unopened. No
second grid extension is allowed even if the selected value is again a
boundary. `SCALAR_SUFFICIENT` still requires both paired-bootstrap gains above
zero, at least 20% of sender headroom recovered, and a non-boundary optimum.

### Per-channel affine rung (frozen before execution)

Because the finalized scalar rung closes only 2.8146% of headroom, the next
map is `z = scale * LayerNorm(h_sender) + bias`, with 768 learned scales and
768 learned biases. Both vectors start at zero: the visible injection is
exactly neutral, while both have a direct gradient path. Sender and receiver
remain frozen.

- Data: the same hash-defined TinyStories buckets; first 64 eligible train
  documents and the same 32 dev documents. Test is neither tokenized nor
  scored.
- Optimization: Adam, learning rate `1e-2`, batch size 8, at most 30 epochs,
  deterministic epoch shuffles, dev-NLL early stopping with patience 5. No
  learning-rate or epoch retry is permitted after observing dev.
- Required dev arms: neutral; correct sender state; the same learned map with
  sender states rotated across documents; and a separately optimized same-
  shape map trained with sender vectors forced to zero.
- Evidence: per-document NLL/top-1, paired 10,000-sample bootstrap intervals,
  fraction of sender headroom closed, finite checks, deterministic parameter
  fingerprint and evaluation replay.
- `PER_CHANNEL_SUFFICIENT` requires correct-state NLL to beat neutral,
  shuffled-document, and zero-sender controls with all paired 95% intervals
  above zero, and to close at least 20% of sender headroom. Otherwise advance
  only to a preregistered low-rank rung; do not run dense.

## Mechanism branch: KV-cache handoff (distinct from the embedding-injection bridge above)

The embedding-injection ladder above (`z` prepended to receiver input embeddings)
was followed by a separate, parallel mechanism: transferring the sender's
**KV-cache** directly (`past_key_values`), corrected by a small shared
low-rank adapter (`LowRankKVAdapter`: per-selected-layer rank-8 additive
correction to K and V, sender layers `(1,3,5,7,9,11)` mapped onto the
receiver's 6 decoder layers). This is a different transport (cache, not a
token embedding) and is tracked separately; results here do not carry over
to the embedding-injection arms above, and vice versa. Established findings
on this branch, in brief: raw (untrained) KV transfer already carries
information (`KV_CACHE_RAW_TRANSFER_SUPPORTED`); a trained low-rank adapter
carries a single fact/2-token bandwidth and a 32-token global sequence
pattern (`MINIGUNPOINT_GLOBAL_PATTERN_TRANSFER_SUPPORTED`, test acc 96.9%);
but three different adapter designs for 2-key addressing, and one XOR-style
2-fact compositionality task, all returned `NOT_SUPPORTED`, confounded at
the time by an optimization-regime defect (see below).

### Optimization-regime confound and its fix (frozen finding)

Every KV-branch and embedding-injection training loop up to this point that
was NOT already using a full-batch step (`phase1_controlled_fact_transfer_2token.py`
and several capacity-ladder scripts: `dense_no_bias`, `low_rank_sweep`,
`diagonal_scale_only`, `latent_prefix_tokens` (both variants),
`full_passthrough_trained_translator`) used a per-example loop -- one
optimizer step per single training document, LR fixed at `1e-2` -- rather
than one gradient step per epoch over the whole (64-document) train set.
A dedicated diagnostic (`phase1_controlled_fact_2token_lr_diagnostic.py`,
64 train / 32 dev, full-batch, LR in `{3e-3, 1e-2, 3e-2}`, 200-epoch cap,
one seed) showed that switching to full-batch training with LR=`1e-2`
produces a real, bootstrap-CI-confirmed correct-vs-shuffled separation
(10.10 nats, CI `[9.05, 11.18]`) where every prior per-example-loop run at
the same LR had been dead (0%/0% top-1, no separation). Verdict:
`SUPPORTED` (optimization-regime confound confirmed) -- but scoped
**only** to the scripts listed above that used the per-example loop. The
2-key addressing and compositionality scripts were *already* full-batch at
LR=`1e-2` with a larger step budget (300 and 100 epochs respectively), so
this confound does not reopen their `NOT_SUPPORTED` verdicts; those stand.
Going forward, every KV-branch script uses full batch (one step/epoch over
the whole train set) and MPS by default.

### SQL-lookup key-value binding experiment (run, not merely proposed)

Motivation: `MINIGUNPOINT_GLOBAL_PATTERN_TRANSFER_SUPPORTED` only shows the
channel carries a *global* sequence-level signal; it says nothing about
whether the channel can do associative lookup -- address one specific fact
among several by key and recover the bound value. This is the cleanest
available addressing+binding instrument: a 5-row `name -> id` table (`SELECT
id WHERE name = ?`, except the row store is a KV-cache).

**Protocol** (`phase1_kv_cache_sql_lookup_binding.py`, frozen before
execution): 5 names drawn from a fixed, tokenizer-verified
single-BPE-token pool (`Bob, Alice, Ivan, Lena, Anna` -- `Masha` from the
original proposal does not tokenize to 1 token and was substituted);
5-digit ids, freshly and uniquely bound to the 5 names per document
(binding is a random bijection, regenerated per document), verified to
tokenize to a constant 2 BPE tokens. Per document, the table's on-page
**display order** of the 5 lines is independently reshuffled, so line
position cannot leak identity. The receiver sees only `What is {name}'s
ID?` for one of the 5 names chosen uniformly at random, plus the
transferred (adapted) KV-cache -- never the table text itself. Train=64 /
dev=32 / test=32 documents, disjoint by exact document key (hash of
binding+order+query), full-batch training (one step/epoch over all 64),
LR=`1e-2`, Adam, dev-NLL early stopping (patience 10 dev-evals = 50
epochs), MPS. Arms: `correct` (own adapted KV); `shuffled_document`
(another document's adapted KV); `wrong_binding` (the *same* 5 id values,
reassigned to names by a fixed rotation/derangement -- no name keeps its
true id -- rendered as its own fresh table text and re-encoded by the
frozen sender, so only the name<->id association changes, not the display
order or the value set); `zero_sender`; and a secondary sanity arm,
`query_name_swap` (the document's own correct KV, queried about a
*different* true name from the same table). Primary gate: `correct` must
beat both `shuffled_document` and `wrong_binding` on a paired per-document
bootstrap CI, evaluated once on the test split, which is not tokenized or
scored until training and early stopping are complete.

**Result** (single run, converged by early stopping at epoch 80,
`epochs_capped=False`, wall time 30.5s, test N=32, all four bootstrap
comparisons finite):

| arm | exact accuracy | NLL |
|---|---|---|
| correct | 62.5% | 1.447 |
| shuffled_document | 3.1% | 15.386 |
| wrong_binding | 3.1% | 6.776 |
| zero_sender | 0% | 10.462 |
| query_name_swap (secondary) | 71.9% | 1.605 |

Paired bootstrap 95% CIs (`correct` minus other, 10,000 resamples): vs
shuffled `[11.999, 15.731]`; vs wrong_binding `[3.913, 6.778]`; vs zero
`[8.196, 9.724]` -- all exclude zero. `wrong_binding` is rejected as an
alternative explanation. Per-token top-1 breakdown shows the effect is
concentrated on the first (identity-bearing) of the 2 target tokens
(correct 68.75% vs wrong_binding 3.1%); the second token is highly
predictable from the first regardless of binding (correct 93.75% vs
wrong_binding 96.9%), consistent with BPE digit-continuation structure
rather than a scoring artifact.

**Verdict: `SQL_LOOKUP_BINDING_SUPPORTED`.** Scope: one seed, one run, N=32
test documents, 5-name/5-digit-id/2-token-target regime, GPT-2->DistilGPT2
pair, rank-8 shared low-rank KV adapter over 6 fixed layers. Does not by
itself establish robustness (untested: more names/rows, multi-seed
replication, held-out name pool, adversarial/high-confidence wrong-binding
distractors, or whether this survives the same addressing stress this
adapter design already failed on the earlier abstract 2-key task -- the
difference in outcome between that `NOT_SUPPORTED` result and this
`SUPPORTED` one is itself unexplained and worth isolating before
generalizing either way).
