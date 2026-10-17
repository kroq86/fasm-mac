# Related work — formulations we can borrow

PDFs in this directory (downloaded 2026-09-22):

| File | Paper |
| --- | --- |
| `2608.13317.pdf` | StateBridge |
| `2605.11167.pdf` | Bicameral Model |
| `2608.16347.pdf` | Architecture-Dependent Causal Transfer |
| `2606.05711.pdf` | Beyond Tokens (survey) |
| `2309.16042.pdf` | Zhang & Nanda — activation patching best practices |
| `2403.00745.pdf` | AtP* (component localization; **not** the primary MI patching cite) |

Use **Zhang & Nanda (2309.16042)** for activation patching as method. AtP* is a different question (efficient attribution), not “activations as communication.”

---

## Borrowable vocabulary (safe, cite them)

### From *Beyond Tokens* (Liu, 2606.05711)

**Use as taxonomy scaffold:**

> Latent communication transfers continuous model state—embeddings, hidden states, or KV caches—directly between agents.

Three axes (quote/paraphrase with cite):

- **WHAT** — transmitted representation  
- **WHICH** — sender–receiver spaces / layers aligned  
- **HOW** — how the received state is fused into the receiver  

**Our placement:** WHAT = final-layer / residual activation; WHICH = same GPT-2 family, **no** learned alignment; HOW = **localized same-site replace** (vs continuous add / prefix).  
This is the cleanest framing for novelty: we are a **HOW** result, not a “first latent message” claim.

Also useful contrast language from the survey:

> sender decode → token transport → receiver encode (natural-language pipeline)

vs latent-comm. Parent KV-handoff paper already lives in this space; residual extension is another WHAT/HOW cell.

### From *StateBridge* (Peng et al., 2608.13317)

**Borrow the problem statement, not their solution:**

- “**discrete bottleneck**” — text forces continuous → tokens → embeddings.  
- Matching dimensionality is **not** sufficient; final-layer HS ≠ input embedding geometry.  
- Prior split they name: (i) layer-by-layer / KV injection, (ii) **trained projectors**.  
- Their gap: can mismatch be fixed by **alignment alone**, training-free?

**Their HOW:** closed-form **orthogonal Procrustes** + norm calibration + vocabulary anchoring → **continuous prefix**.

**Our contrast sentence (honest):**

> Unlike StateBridge, which *aligns* sender final-layer states into the receiver’s input embedding space and prepends them as a continuous prefix, we leave the foreign top-layer activation **unaligned** and ask whether **localized causal-site replacement** alone can make it a usable operand for a pretrained computation—on same-family GPT-2 copies.

Do **not** claim StateBridge “failed without alignment”; they never tested our patch protocol. Do claim: different HOW, different obligation (task scores vs \(f(X)\) ladder).

### From *Bicameral* (Flamant et al., 2605.11167)

**Borrow the high-level question:**

> Can two pretrained language models coordinate through a continuous channel instead of serializing every exchange through the output vocabulary?

**Their mechanism:** frozen LMs + **trainable** translation network + **learned suppression gate** (~1% params), lockstep bidirectional.

**Our contrast:**

> Bicameral shows that *learned* hidden-state coupling can support tool-augmented coordination. Our Step 5 asks a stricter training-free question: can a foreign activation act as an operand **without** a task-trained interface, when routing (HOW) is fixed?

Useful residual-stream indexing language (layer \(\ell\) = residual before block \(\ell\)) — already compatible with our native notes.

### From *Architecture-Dependent Causal Transfer* (Cárdenas, 2608.16347)

**Most important methodological vocabulary — use this:**

1. Three levels (cite as their ladder, map to ours):  
   - representational similarity  
   - cross-model retrieval from projected states  
   - **end-to-end causal transfer via activation injection**

2. Critical sentence (paraphrase + cite):

> Representational alignment does **not** by itself establish that a similarity signal can be exploited — projected onto a target and shown to **causally** change behaviour.

3. Scope language they insist on (gold for honesty):

> causal transfer of the **representational vehicle**, not of **meaning**

We can say: their end-to-end success is **vehicle-level** and architecture-dependent; our positive Step 5 is stronger *when it holds* (content-specific \(f(X)\)), but **narrower** (same GPT-2×2, category map, patch HOW), and Step 6 (new \(f(X)\) state) is REFUTED—so we still do **not** claim “shared understanding” or relay.

4. Method detail worth echoing: activation patch inside a forward loop (replace hidden at a site) rather than broken generate()+cache hacks — same family as our native same-site patch.

### From Zhang & Nanda (2309.16042)

**Borrow:** activation patching as **causal** localizer between clean/corrupt (or donor/receiver) runs; metrics matter; patching ≠ interpretability of meaning by itself.

**Our distinction:**

> Standard activation patching is usually an *analysis* tool within one model (or matched runs). We use same-site replacement of a **foreign** model’s activation as a **communication primitive**, then demand pretrained \(f(X)\) use—not only a logit flip.

---

## Claim wording comparison

| Too weak / already owned by others | Prefer |
| --- | --- |
| “Hidden states can be transferred between LLMs” | (StateBridge, Bicameral, survey, Causal Transfer already occupy this) |
| “Training-free latent communication” | StateBridge owns this phrase for *aligned prefix* |
| “Causal activation transfer” | Causal Transfer owns vehicle-level cross-arch + learned proj |
| | **Foreign top-layer activation from one frozen LM can act as a functional operand for another frozen LM’s pretrained computation, without a learned translator, when routed to the correct causal site** (same-family; evidence ladder 1–5; Step 6 closed) |

### Phrases we should **not** steal as if ours

- “StateBridge” / “continuous prefix” / “vocabulary anchoring” / “Procrustes alignment” as our method names  
- “Bicameral” / “suppression gate”  
- “Platonic Representation Hypothesis” as our conclusion  
- Claiming AtP* as our MI method paper  

### Phrases we **should** cite when using

- discrete bottleneck (StateBridge / Beyond Tokens)  
- WHAT / WHICH / HOW (Beyond Tokens)  
- vehicle vs meaning (Causal Transfer + philosophy cites they use)  
- training-free vs trainable interface (StateBridge vs Bicameral contrast)  
- activation patching metrics/methods (Zhang & Nanda)

---

## Suggested Related Work paragraph (draft English)

Paste into the extension / parent appendix:

> Latent communication surveys now organise the area along three axes—WHAT is transferred, WHICH spaces are aligned, and HOW the state is fused [Beyond Tokens]. Training-free methods such as StateBridge align final-layer sender states into the receiver input space (orthogonal Procrustes, norm/vocabulary calibration) and inject them as a continuous prefix [StateBridge]. Learned interfaces such as the Bicameral Model couple two frozen LMs bidirectionally through a translation network and gate [Bicameral]. Architecture-Dependent Causal Transfer separates representational similarity, retrieval, and causal injection, and explicitly scopes positive causal results to transfer of a representational *vehicle* rather than meaning, finding architecture dependence under learned projections [Causal Transfer]. Activation patching provides the causal replacement tool [Zhang & Nanda]. Our extension sits in this map as a same-family, **translator-free** cell whose positive result is a **HOW** claim: raw foreign top-layer activations become usable operands for a pretrained category map under localized same-site replacement, after an evidence ladder that separates dynamics, decode, causal basin effects, identity lifetime, and functional \(f(X)\); a subsequent test that use creates a new transferable \(f(X)\) state fails.

---

## Gap list (still needed for a hard paper claim)

From the external note and these papers’ standards:

1. **Confounds:** norm / prior / common-mode (Causal Transfer + our continuous sentiment failure already warn).  
2. **≥2 functions \(f\)** (category alone is thin vs StateBridge’s multi-task eval).  
3. **Site map** (HOW is the claim → must show site sensitivity).  
4. Expanded \(n\) (note: some drafts cite 35/36, mean \(S\approx0.53\), \(t\approx15.5\) — only promote after ledgered artifact; current ledger still has 6/7, \(S=+0.563\), \(t=4.34\)).  
5. Do not overclaim vs StateBridge without a matched “raw L12 + prefix” negative control on the same \(f\).

---

## Numbers policy

Until an artifact is folded into `evidence.md` / `evidence.json`, the extension’s primary Step 5 number remains **6/7, mean \(S=+0.563\), \(t=4.34\)**. Treat 35/36 as a candidate expansion, not manuscript truth.

---

## KV-cache transfer papers (HTML downloaded 2026-10-17, `related/html/`)

Used in the conference slides (`papers/kv-handoff/slides.html`). Paraphrase with citation; do not copy sentences.

| arXiv | Paper | Borrowable framing / number |
| --- | --- | --- |
| 2510.03215 | Cache-to-Cache, Fu et al. 2025 | Text is a low-bandwidth "information bottleneck": internal representations compressed into token strings and decompressed by the receiver; token-by-token latency. C2C: +3.1–5.4% accuracy vs text communication, ~2.5× lower latency. |
| 2605.22863 | Latent Cache Flow, Rossi et al. 2026 | Text channel is "slow" (receiver waits for decode) and "lossy" (state compressed into discrete tokens). C2C requires identical context; LCF adapter 13 MB vs C2C 956 MB. |
| 2608.03893 | Cross-Model KV Cache Transfer, Heo et al. 2026 | "Each swap forces the receiver to repay the prefill"; "prefix caching mitigates this within a single model only". Closed-form ridge mapper within family: 73–98% retention on 4 of 6 pairs, 2 degrade sharply; 2.7–25× faster than re-prefill. Consistent with our linear-translator result. |
| 2608.04893 | When Does Latent Communication Pay?, Cheng et al. 2026 | "Benchmark deltas do not by themselves establish latent-thought transmission"; needs a mismatched-cache audit (deranged / zeroed / moment-matched). "A large cache effect need not be a pairing effect" (zeroing −14.7 pts, mismatch −0.4). C2C's released projector: no detected example-specific transfer. Same idea as our swap test. |
| 2608.20617 | XKV, Liu et al. 2026 | Text "places autoregressive decoding on the critical path"; heterogeneous frozen pairs need a trained translator (C2C, LCF-X, XKV). |
