# Extension: Evidence ladder for training-free residual routing

**Status:** draft material for expanding the KV-handoff case study — **not a separate paper**.  
**Parent manuscript:** [`paper.md`](paper.md).  
**Evidence ledger:** [`../latent-transport/evidence.md`](../latent-transport/evidence.md).

---

## Why this extension exists

The parent paper asks whether a transferred internal state is *used*, not merely present or perturbative. That question was studied there for **KV-cache adapters** (GPT-2 → DistilGPT2).

Here we continue the same evaluative discipline on a different interface: **residual / hidden-state injection** between two frozen GPT-2 124M copies, with **no learned translator** on the positive protocol. The engineering intuition is a latent API; the scientific obligation is a ladder. A positive Step 5 alone is not enough: if Steps 2–4 are soft, Step 5 can look like “semantic use” while really being decode leakage, nonspecific injection, or a transient basin flip. Conversely, Steps 1–4 can succeed while Step 5 fails—exactly what continuous residual flooding showed.

**Parent vs this extension**

| | Parent | This extension |
| --- | --- | --- |
| Payload | KV cache + rank-8 adapter | Residual / final-layer hidden state |
| Models | GPT-2 → DistilGPT2 | GPT-2 → GPT-2 (copies) |
| Training | Task-specific adapters | None for the positive patch protocol |
| Question | Does the receiver *use* transferred information? | Same, different HOW |

---

## Architecture under test (three protocols — do not conflate)

All runs: native engine (`tensor_gpt2_handoff.h` family), frozen GPT-2×2, separate KV caches, no shared cache, no MLP / cross-attn substitute bridge.

| Protocol | Mechanism | Ladder role |
| --- | --- | --- |
| **Leaky+tanh \(z\)** | Minimal channel state \(z\), continuous loop | Steps 1–4 (dynamics, decode, causal WTA, identity dwell) |
| **Continuous \(\alpha h\)** | Weak additive residual (\(\alpha \approx 0.02\)) | Failed / inconclusive **functional** Step 5 |
| **Same-site activation patch** | Full foreign residual, one shot, fixed \((L,P)\) | Positive **functional** Step 5; Step 6 null |

Verdicts for Steps 1–4 are about the leaky channel’s information properties. The positive Step 5 is about **routing**: same-site patch, not continuous flooding. Mixing “channel works” with “continuous \(f(X)\) works” is a methodological error.

---

## The ladder (why each rung exists)

Order is load-bearing. Each step blocks a cheaper failure mode of the next:

```text
1 dynamics          →  without this, “communication” is just noise in the harness
2 decode            →  without this, interventions can only move residuals nonspecifically
3 causal content    →  without this, decode is spectator knowledge, not receiver effect
4 identity + dwell  →  without this, Step 3 may be category/magnitude, not a specific operand
5 functional f(X)   →  without this, Steps 1–4 are interesting dynamics, not latent computation
6 new content f(X)  →  without this, no relay / return path (Steps 7–8 blocked)
```

A late positive result does **not** retroactively prove earlier rungs. Each rung has its own hypothesis, control, and stop rule (`SUPPORTED` / `REFUTED` / `INCONCLUSIVE`). No α/β/λ retuning after INCONCLUSIVE/REFUTED to force the next step.

---

## Plain claim of the positive apex (Step 5, patch protocol)

One frozen GPT can hand another an internal hidden state such that the second treats it as a **meaningful operand** for a computation it already knows—not as noise, not as a mere strong signal.

- \(A\) processes `dog`. No token `"dog"` is sent to \(B\). No text. No trained translator.
- \(A\)’s internal vector is inserted at a **specific causal site** in \(B\).
- \(B\) applies pretrained knowledge: \(f(\mathrm{dog})=\mathrm{animal}\).

Formally:

\[
A{:}\;\mathrm{hidden}(\mathrm{dog})
\;\xrightarrow{\text{localized patch}}\;
B
\;\xrightarrow{f}\;
\mathrm{animal}
\]

without \(A \to\) token \(\to\) text \(\to B\).

Ordinary API: `token → GPT → token`.  
Shown possibility: `foreign hidden → GPT → meaningful pretrained computation`.

Non-trivial because same width ≠ same meaning. Continuous weak injection often produced common-mode logit moves (“something changed”) without content-specific \(f(X)\). Correct-site full patch did.

---

# STEP 1 — Coupling changes dynamics

**Claim.** Connecting \(A\) and \(B\) changes trajectories relative to matched uncoupled baselines.

**Why this step.** If ON and OFF look the same, later “effects” are measurement theater.

**Hypothesis.** ON coupling alters \(|h|\), basins, or WTA structure vs always-OFF / coupling-OFF under the same tokens.

**Experiments / artifacts.**
- `native_self_excitation_onoff.c` → `/tmp/onoff.log` (scalar \(h' = h + \alpha h_{\mathrm{other}}\); ON for \(t<15\), then OFF).
- `native_leaky_ablation.c` → `/tmp/leaky_ablation.log` (Ablation 2: permanent ON vs OFF).
- `native_attractor_basin.c` → `/tmp/attractor_basin.log` (kick-dependent basins).
- WTA matrix: `native_winner_take_all.c`.

**Controls.** Always-OFF baseline; identical filler drive; symmetric WTA control.

**Results (native).**
- Leaky Ablation 2, \(\lambda=0.8\): OFF \(|h_a|\) at \(t=10/30/59\) = 479 / 422 / 458; ON = 535 / 449 / 484 with nonzero \(|z|\).
- ON/OFF, \(\alpha=0.05\): during ON, \(|h_b|\) spikes to ~1555 while always-OFF stays ~450–550.
- Basins: across kicks, \(|h_a|\) spans ~415–1132; \(|z|\) stays in a tight attractor (std/mean ~5%).

**Verdict: SUPPORTED.**

**Does not prove.** Information transfer, causal content use, or \(f(X)\). Instability ≠ communication.

---

# STEP 2 — Channel carries information (decode)

**Claim.** Channel state contains sender content: \(P(\mathrm{content}\mid z) >\) chance. Observational only.

**Why this step.** Without decode, Step 3 interventions cannot be attributed to *content*. Without a held-out split, “decode” can be memorizing the training kicks.

**Hypothesis.** Nearest-centroid family readout on held-out kicks exceeds chance.

**Experiment.** Leaky+tanh (\(\lambda=0.8\), \(\alpha=0.05\), \(\beta=0.4\)). Centroids from `/tmp/z_vectors_10.txt`; score `/tmp/z_vectors_holdout.txt` via `native_attractor_holdout_probe.c`. Binary digit vs word family. (Auxiliary: Pearson(\(\mathrm{emb\_cos}\), \(z\_\mathrm{cos}\))=0.573 in `/tmp/run_corr2.log`.)

**Controls.** Disjoint held-out identities; centroids frozen before scoring; chance = 50%.

**Result.** Held-out **10/10 = 100%** (chance 50%). Odd digits → digit; Piano/Forest/Bridge/Garden/Castle → word.

**Verdict: SUPPORTED.**

**Does not prove.** That \(B\) *uses* the content; causality; identity at chance levels finer than family (that is Step 4); functional \(f(X)\).

**Caveat for the paper.** This is a **binary family** probe, not 10-way identity. That is intentional: Step 2 only needs “content in \(z\)”; fine identity is Step 4. A soft Step 2 would invalidate attributing Step 5 to transfer of *that* content—but 10/10 held-out is not soft.

---

# STEP 3 — Channel information causally affects the receiver

**Claim.** Changing transmitted content changes receiver behavior in the direction predicted by that content. Intervention, not correlation.

**Why this step.** Decode can be true while \(B\) ignores \(z\). Real vs matched control separates content-specific causation from “any injection moves basins.”

**Hypothesis.** Hijacking \(B\)’s inject with FOREIGN \(A\)-trajectory flips \(B\) toward FOREIGN more than a mismatched control hijack.

**Experiment.** `native_wta_intervention_full.c`, \(\alpha=0.07\), \(n=260\). REAL: matched-column FOREIGN. CONTROL: mismatched `b_ctrl`.

**Result** (`/tmp/wta_full_run.log`):
- REAL flip-to-FOREIGN: **163/260 = 62.7%**
- CONTROL scored flip: **125/260 = 48.1%**
- stay-TRUE: 37.3% real vs 51.9% control

**Verdict: SUPPORTED.**

**Does not prove.** Semantic category/sentiment use; long-lived identity; multi-step reasoning. This is basin / WTA arbitration under leaky coupling—not yet \(f(X)\).

---

# STEP 4 — Specific identity + lifetime

**Claim.** The channel carries a **specific** identity (not only family/magnitude), with measurable post-splice lifetime.

**Why this step.** Step 3 could be “foreignness” or energy. An operand for \(f(\mathrm{dog})\) needs a dog-like identity, not a blob. Dwell shows whether that identity is a usable object or a one-tick flash.

**Hypothesis.** One-shot splice of a digit centroid into the live loop is 10-way classifiable above chance immediately, then decays; matched-norm random splice stays near chance.

**Experiment.** `native_z_identity_confusion.c`: splice at \(T=40\); classify vs 10 centroids; dwell \(\in\{0,1,2,5,10,20\}\); control = same-norm random. \(\lambda=0.8\), \(\alpha=0.05\), \(\beta=0.4\). Chance = 10%.

**Result** (Claude native run; matches preregistered pattern):

| dwell | real | control |
|------:|-----:|--------:|
| 0 | 100% | 0% |
| 1 | 60% | 10% |
| 2 | 20% | 10% |
| 5 | 10% | 20% |
| 10 | 10% | 10% |
| 20 | 10% | 10% |

**Verdict: SUPPORTED** (transient identity; lifetime ≈ 1–2 steps).

**Does not prove.** Semantic \(f(X)\); persistent memory; that continuous residual flooding can feed pretrained circuits.

---

# STEP 5 — Immediate functional use \(f(X)\)

**Claim.** \(B\) uses transferred \(X\) as an argument to a **pretrained** computation \(f\), not merely shifts basins or common-mode logits.

**Why this step.** This is the first claim that a foreign hidden state is a **latent API operand**. Steps 1–4 can all pass while this fails.

Stronger than: “info about dog is in the vector”; “stuffing it changes \(B\)”.

Weaker than (and not claimed): multi-hop relay, training-free composition, cross-family transfer.

---

## Step 5 — continuous residual path (failed / stopped)

**Protocol.** Weak continuous \(\alpha h\) into residual (\(\alpha \approx 0.02\)). Template e.g. “A thing is a type of”; \(X\) only via channel.

### 5-cont-A — category

**Metric.** Golden leave-one-out cross-category \(S\) (matched foils).

**Result** (`native_step5_category.c`): mean \(S \approx +0.126\), std \(\approx 0.187\), \(S>0\): 4/7, \(t \approx 1.78\), \(p \approx 0.13\). Within animals: dog/cat negative, horse positive.

**Verdict: INCONCLUSIVE** (not positive).

### 5-cont-B — sentiment

**Result** (`native_step5_sentiment.c`): large common-mode \(\Delta\)pos ≈ \(\Delta\)neg (e.g. great: +3.844 / +3.851). Mean differential \(S \approx +0.0069\), \(t \approx 0.77\).

**Verdict: REFUTED** for polarity use under this operation — magnitude without semantic differential.

**(+1 induction):** invalid instrument (needs real token position); not counted as architecture REFUTED.

**Interpretation.** Information can be decodable and causally active (Steps 2–4) while continuous flooding fails as \(f(X)\). Common-mode logit shifts are **not** semantic transfer.

---

## Step 5A / 5B — mechanism (routing, not “alignment as slogan”)

Early continuous failure invited a “representation mismatch” story (top \(A\) → early \(B\)). User correction: do **not** treat naive X-absent same-layer inject as the diagnosis, and do **not** call global alignment REFUTED from that alone.

**Decisive contrast:** continuous weak add vs **localized one-shot same-site replace**.

### Native same-site ceiling (`native_activation_patch.c` → `/tmp/activation_patch_ext.log`)

DOG pos-1 activation patched into ROSE run at same \((L,P)\):

- Baseline gap (animal−plant): DOG \(+3.61\), ROSE \(-1.69\).
- Patch at \(L=0\): gap \(\to +3.614\) (\(\Delta\) vs ROSE \(+5.309\)).
- Self-patch ROSE→ROSE: \(\Delta = 0\) (sanity).
- Reverse ROSE→DOG: gap flips toward plant.

So same-site residual **can** carry category-relevant causal content inside one model.

### Foreign \(A_{L12}\) same-site category (primary Step 5)

**Protocol.** Isolated \(A_{L12}(X)\); full replace at \(L=0\), \(P=1\); no \(\alpha\) tuning; seven members; golden leave-one-out \(S\).

**Result:**

| member | target | \(S\) |
| --- | --- | ---: |
| dog | animal | +0.693 |
| cat | animal | +0.712 |
| horse | animal | +0.584 |
| rose | plant | −0.127 |
| oak | plant | +0.322 |
| hammer | tool | +0.720 |
| wrench | tool | +1.038 |

- \(S>0\): **6/7**
- mean \(S = +0.563\), std \(0.343\), \(t = 4.34\) (df=6), \(p \approx 0.005\)

Animals and tools move content-specifically; one plant (rose) is the miss. \(B\) is not merely reacting to signal strength.

**Foil-gap corroboration** (`/tmp/activation_patch_7pair.log`): raw bridge \(S_{\mathrm{br}}\) mean \(+3.57\) is inflated by common-mode vs SELF/SHUF (~+2.9 / +3.1); residual content-specific differential remains after that audit. Prefer golden \(S=+0.563\) as the paper number.

**Verdict (functional Step 5, patch protocol): SUPPORTED.**

**Does not prove (at the time this was first written).** Continuous flooding works; Step 6 (new \(f(X)\) state); return path; cross-family / DistilGPT2; that every site works; whether \(n=7\) survives scale; whether the effect is explained by norm/prior/common-mode rather than content; whether it generalizes past member→category. The three scale/confound/generalization gaps are closed below (adversarial verification pass). Cross-family, site map, and cross-checkpoint remain open.

**Routing formulation (scoped):** functional latent communication is, in this regime, a **routing problem**: foreign activations become usable when delivered to the correct causal site. Continuous flooding ≠ localized patch.

---

## Adversarial verification pass on Step 5 (n=36 scale, confound controls, second function family)

Run under an explicit adversarial-verifier mandate: the goal was to break the 6/7 result, not to make it prettier. Protocol, causal site \((L{=}0, P{=}1)\), and golden-\(S\) formula (strict cross-category leave-one-out) held fixed throughout — no \(\alpha\)/temperature/prompt tuning after seeing any run. Each sub-experiment was preregistered (dataset frozen from bare-GPT-2 native screening *before* any latent run) and reported once, per the calibration-first discipline: no automatic retry after an INCONCLUSIVE/REFUTED sub-result.

### Part A — scale from \(n=7\) to \(n=36\)

Five categories (animal, plant, tool, vehicle, furniture — all single-token category words). 42 single-token member candidates screened on bare GPT-2 with the same template ("A \(X\) is a type of"), margin \(\ge 1.0\) nat fixed in advance; 36 passed and were frozen to `/tmp/step5_scaled_set.txt` before any latent run.

**Result** (`/tmp/step5_scaled_n36.log`):

| | value |
| --- | ---: |
| \(S>0\) | **35/36 (97.2%)** |
| mean \(S\) | **+0.5312** |
| std | 0.2052 |
| \(t\) (df=35) | **15.53**, \(p \approx 10^{-16}\) |
| per-category | animal 12/12, tool 5/5, vehicle 10/10, furniture 4/4, plant 4/5 |

Only miss: **rose** (\(S=-0.157\)) — the same word that was the sole miss in the original \(n=7\) run, same sign, similar magnitude. Treated as a natural failure case, not patched or excluded.

**Verdict: SUPPORTED**, with a much larger margin than the \(n=7\) pilot.

### Part D — norm / prior / common-mode / generic-perturbation controls

Attacked the three cheapest alternative explanations for Part A, using the frozen \(n=36\) set and site, no new tuning (`/tmp/step5_scaled_D_controls.log`, raw data in `/tmp/step5_scaled_raw.csv`, `/tmp/step5_scaled_random_control.csv`):

- **D1 — norm correlation.** \(\mathrm{corr}(S, \lVert\text{injected vector}\rVert) = +0.002\). Effect size has no relationship to injection magnitude.
- **D2 — random vector, matched norm**, patched in place of the real content, scored against the *same* real leave-one-out control. \(S_{\text{random}}>0\): 22/36 (61%, chance-level for this metric), mean \(=+0.081\), \(t=0.50\) — collapses from \(t=15.53\) to noise. A generic perturbation of identical magnitude does not reproduce the effect.
- **D3 — permutation control.** Same real logits, category labels shuffled, 200 trials. mean\((S{>}0\) count\()=15.6/36\) (chance for 5-way), mean(mean \(S\))\(=-0.011\), and **0/200** shuffles reached the real mean \(S=+0.531\) (best shuffle: \(+0.173\), \(\approx\)3× below real). Empirical \(p<1/200\).

**Verdict: all three candidate explanations REFUTED.** The effect is not norm, not generic perturbation of matching magnitude, and not an artifact of the scoring formula under label reassignment.

### Part B — generalize beyond member→category

Deliberately chose a structurally different pretrained relation — **object → typical property** (not taxonomy/superclass completion) — *before* running any latent test, to avoid selecting a function because it already looked promising. Screened 18 single-token object candidates across 5 property words (cold, hot, sweet, sour, dark) on bare GPT-2 with template "Everyone knows that \(X\) is", margin \(\ge 1.0\) fixed in advance. Only **9/18** passed; **hot and sour got zero passing members** — reported as-is, not fixed by re-templating. Frozen set: snow/winter/frost→cold, sugar/honey/candy/syrup→sweet, night/shadow→dark.

Same transport protocol (isolated single-position layer-12 activation, one-shot patch at the object's own natural position in this template, layer 0), same golden-\(S\) leave-one-out formula restricted to the 3 populated groups.

**Result** (`/tmp/step5_partB_property.log`):

| | value |
| --- | ---: |
| \(S>0\) | **8/9** |
| mean \(S\) | **+0.1894** |
| std | 0.1410 |
| \(t\) (df=8) | **4.03**, \(p \approx 0.004\) |
| per-group | cold 3/3, sweet 3/4 (syrup \(S=-0.019\), small), dark 2/2 |
| \(\mathrm{corr}(S,\lVert h\rVert)\) | +0.022 |
| random-vector control | \(S_{\text{random}}{>}0\): 4/9, mean \(=-0.061\) (real \(+0.189\)) |

Effect size is smaller than member→category (\(+0.189\) vs \(+0.531\)) — noted honestly, not smoothed over — but sign, significance, and control behavior (random vector collapses the effect) replicate.

**Verdict: SUPPORTED.** Per the preregistered stop rule, no second function family was run after this result.

### What the adversarial pass changes

Three of the four gaps flagged when this section was first drafted are now closed: scale (35/36, \(n=36\)), confound controls (norm/random/permutation all REFUTED as explanations), and a second, structurally different \(f\) (object→property, SUPPORTED). **Site map (does the effect localize to specific \((L,P)\), or is it broad?) remains open** and should be preregistered as a fixed grid before running — no adaptive search for a better site.

---

# STEP 6 — Transform creates new content \(f(X)\)

**Claim (only if Step 5 SUPPORTED).** After \(B\) computes \(f(X)\), \(B\)’s state represents **new** content \(f(X)\), not merely \(X\).

**Why this step.** Step 5 shows \(B\) *uses* \(X\). Relay needs the *result* to become a transferable object.

**Experiment.** Full 7-pair soft-serialization test (Claude; \(T=0.3\) pre-fixed, not tuned). Same golden \(S\) readout aimed at \(f(X)\) identity.

**Result:** \(S>0\): 3/7; mean \(S = -0.0096\); std \(0.832\); \(t \approx -0.03\). Within-category signs conflict (e.g. dog \(+0.82\), cat \(-0.84\)).

**Verdict: REFUTED.**

**Policy.** Branch closed. Do not retune to revive. Steps 7–8 (return path; second computation \(g(f(X))\)) **NOT PURSUED**.

---

# Steps 7–8 — blocked

| Step | Claim | Status |
| --- | ---: | --- |
| 7 | \(B\) returns \(f(X)\) to \(A\) | NOT PURSUED (needs Step 6) |
| 8 | \(A\) computes \(g(f(X))\) | NOT PURSUED |

---

## Summary table

| Step | Claim in one line | Protocol | Verdict |
| ---: | --- | --- | --- |
| 1 | Coupling changes dynamics vs OFF | leaky / ON–OFF | **SUPPORTED** |
| 2 | \(z\) carries content (held-out decode) | leaky | **SUPPORTED** (10/10, chance 50%) |
| 3 | Content-specific causal effect | WTA hijack | **SUPPORTED** (62.7% vs 48.1%, \(n=260\)) |
| 4 | Specific identity + dwell | splice | **SUPPORTED** (100→60→20→chance) |
| 5-cont | Continuous \(f(X)\) category / sentiment | \(\alpha h\) | **INCONCLUSIVE** / **REFUTED** |
| 5-patch | Foreign hidden as operand for \(f\) | same-site patch | **SUPPORTED** (6/7, \(S=+0.563\), \(t=4.34\)) |
| 5-scale | Same, \(n=36\), 5 categories | same-site patch | **SUPPORTED** (35/36, \(S=+0.531\), \(t=15.53\)) |
| 5-controls | Norm / random-vector / permutation attacks on 5-scale | same-site patch | **all REFUTED as explanations** |
| 5-generalize | Object→typical property (\(n=9\)) | same-site patch | **SUPPORTED** (8/9, \(S=+0.189\), \(t=4.03\)) |
| 6 | New transferable \(f(X)\) state | post-use | **REFUTED** |
| 7–8 | Return / second hop | — | not run |

---

## Why Step 5 is not “maybe Step 2 was wrong”

| Failure mode | What would look like a fake Step 5 | How the ladder blocks it |
| --- | --- | --- |
| No dynamics | Random logit jitter | Step 1 OFF controls |
| Empty channel | Spurious \(S\) from prompt priors | Step 2 held-out 10/10 |
| Nonspecific injection | Any vector flips basins | Step 3 REAL vs CONTROL (62.7 vs 48.1) |
| Only “foreignness” | No dog-specific identity | Step 4 10-way splice + dwell |
| Common-mode magnitude | pos and neg logits move together | Continuous sentiment REFUTED; golden \(S\) contrasts |
| Site / protocol confusion | Continuous fail → “no latent API” | Patch protocol separates routing from flooding |
| Overclaim relay | Treat \(f(X)\) as new message | Step 6 null |

If Step 2 had been chance-level, we would **stop** and refuse a Step 5 “operand” story. It was not. If Step 5-patch had matched continuous sentiment (common-mode only), we would refuse “latent API” and keep at most Steps 1–4. Golden \(S\) is content-differential, including consistent animal/tool signs.

---

## One-line claim (extension apex only)

> An internal state of one frozen GPT can be used directly as a semantic input to another frozen GPT so the second applies knowledge it already has—without text and without a learned translator—when that state is routed to the correct causal site; earlier ladder steps establish that the channel carries and causally moves content, and a follow-up test that use creates a new transferable \(f(X)\) state is negative.

---

## Related work — what we borrow vs what we claim

Full notes and downloadable PDFs: [`related/FORMULATIONS.md`](related/FORMULATIONS.md).

Latent communication is already surveyed along three axes—**WHAT** is transferred, **WHICH** spaces are aligned, and **HOW** the state is fused [Liu, Beyond Tokens, arXiv:2606.05711]. We are not claiming to invent “hidden-state transfer.” Closest cells:

| Work | Closest idea | Their HOW / training | Our contrast |
| --- | --- | --- | --- |
| StateBridge (arXiv:2608.13317) | Training-free latent message; avoid **discrete bottleneck** | Orthogonal Procrustes + norm/vocab calibration → **continuous prefix** | We keep foreign \(A_{L12}\) **unaligned** and succeed only under **localized same-site replace**, not prefix/flooding |
| Bicameral (arXiv:2605.11167) | Two frozen LMs, continuous bidirectional channel | **Trainable** translate + gate (~1%) | Step 5 asks translator-free operand use |
| Causal Transfer (arXiv:2608.16347) | Similarity ≠ retrieval ≠ **causal** injection; scopes success to **vehicle**, not meaning | Learned projection; architecture-dependent | Same ladder spirit; our positive cell is same-family, no projector, content-specific \(f(X)\); Step 6 still REFUTED |
| Zhang & Nanda (arXiv:2309.16042) | Activation patching as causal tool | Analysis within matched runs | We use foreign activation as a **communication primitive**, then demand pretrained \(f(X)\) |

**Borrowable phrases (cite them):** discrete bottleneck; WHAT/WHICH/HOW; vehicle vs meaning; representational alignment ⇏ causal exploitability; training-free vs trainable interface.

**Paper claim (not “latent messaging exists”):**

> A foreign top-layer activation from one frozen LM can act directly as a functional operand for another frozen LM’s pretrained computation, without a learned translator, when routed into the correct causal site.

**Updated after the adversarial verification pass.** Confound controls (norm/prior/common-mode) and ≥1 more \(f\) beyond member→category are now ledgered — see above. Expanded count is ledgered: \(n=36\), \(S=+0.531\), \(t=15.53\); the original 6/7, \(S=+0.563\), \(t=4.34\) is the pilot that motivated it. Still thin for submission until: a preregistered **site map** (does functional transfer localize to specific \((L,P)\)?), cross-checkpoint or cross-instantiation replication, and — separately — a resolution of the naive vs golden \(S\) discrepancy noted in the foil-gap audit (`C5d′`), which should be explained, not just avoided by formula choice.

---

## Suggested placement in the parent manuscript

Late section or appendix after native KV reproduction: **“Extension: training-free residual operands and the evidence ladder (same-family GPT-2)”**. Abstract add-on (one sentence):

> A later native residual-routing study on two frozen GPT-2 copies walks a dynamics→decode→causal→identity→\(f(X)\) ladder: Steps 1–4 supported on a minimal leaky channel; continuous residual flooding fails functional use; localized same-site patching makes a foreign final-layer state a usable operand for a pretrained computation — a 7-pair pilot (mean \(S=+0.563\)) scaled to 36 members across 5 categories (35/36, \(t=15.53\)), survived norm/random-vector/permutation attacks aimed at breaking it, and generalized to a structurally different relation (object→typical property, 8/9); a test that this use creates a new transferable \(f(X)\) state is negative.