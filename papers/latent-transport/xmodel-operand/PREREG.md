# Cross-model operand spike — preregistration (written before any run)

Date: 2026-10-17. Parent result: paper Section 8 Step 5 (GPT-2 -> GPT-2 copy, 35/36, mean S=+0.531).

## Claim X1
A sender GPT-2's final-block hidden state (pre ln_f) for an isolated word, patched with
NO translator into a *different* receiver model's input embedding slot, acts as a
functional operand for the receiver's category completion.

## Protocol (copied from native_step5_scaled.c, re-implemented in PyTorch)
- Sender: GPT-2 124M, word alone at position 0, output of block 12 before ln_f.
- Receiver prompt: "A thing is a type of"; replace the full layer-0 input (token+position
  embedding) at the " thing" position with the patch vector; read last-position logits.
- Categories: animal, plant, tool, vehicle, furniture (category word as receiver token).
- S_i = delta_correct - mean(delta_wrong), delta_c = logit_c(i) - mean over members j of
  OTHER categories of logit_c(j).
- Words: fixed candidate list in spike.py (written before running). Screened per receiver on
  the bare receiver with the native word ("A X is a type of"): keep if correct-category logit
  minus max other-category logit >= 1.0 nat AND the word is a single token for sender and
  receiver. (The original 42-candidate list and screening log were lost from /tmp; this is a
  new set, so absolute S values are not directly comparable to +0.531.)

## Conditions per receiver
1. native: receiver's own embedding of the word (ceiling / sanity).
2. raw: sender GPT-2 state, no translator (the claim).
3. random: Gaussian vector matched in norm to raw, scored against REAL raw leave-one-out.
4. aligned: ridge map sender-final-state -> receiver input embedding, fit on 2000 frequent
   single-token words DISJOINT from the candidate list (StateBridge-like strong baseline).
Also: corr(S_raw, norm), 200-shuffle label permutation on raw.

## Receivers
R0 GPT-2 (same weights, calibration of this re-implementation), R1 DistilGPT2,
R2 Pythia-160m (different family/tokenizer; only if R1 is run cleanly).

## Criteria (fixed now)
- INSTRUMENT VALID only if R0 raw: >=80% S>0 and t>3. Otherwise INVALID -> stop, no tuning.
- X1 SUPPORTED for receiver R: raw >=80% S>0, t>3, random t<2, permutation p<0.01.
- X1 REFUTED for R: raw t<2 or <=60% S>0.
- Otherwise INCONCLUSIVE.
- "Translator-free matters" only if raw is SUPPORTED and aligned is not clearly better
  (aligned mean S <= 1.5x raw mean S). If aligned >> raw, the honest reading is
  "alignment is needed across models", i.e. no translator-free method claim.
- No re-templating, re-sited patching, or threshold changes after seeing results.
Budget: one evening, inference only.
