# Evidence ledger — residual-routing extension (parent: kv-handoff)

Full write-up: [`../kv-handoff/extension-latent-routing.md`](../kv-handoff/extension-latent-routing.md).  
Not a separate paper.

## Claim map

| ID | Step | Statement | Verdict | Primary artifact |
| --- | ---: | --- | --- | --- |
| C1 | 1 | Coupling changes dynamics vs OFF | **SUPPORTED** | `/tmp/onoff.log`, `/tmp/leaky_ablation.log`, `/tmp/attractor_basin.log` |
| C2 | 2 | Held-out channel decode (family) | **SUPPORTED** | `/tmp/z_vectors_{10,holdout}.txt` — 10/10, chance 50% |
| C3 | 3 | Causal WTA content effect | **SUPPORTED** | `/tmp/wta_full_run.log` — 62.7% vs 48.1%, n=260 |
| C4 | 4 | Identity dwell after splice | **SUPPORTED** | `native_z_identity_confusion.c` — 100/60/20 → chance |
| C5a | 5-cont | Continuous category \(f(X)\) | **INCONCLUSIVE** | `native_step5_category.c` — mean S≈+0.126, t≈1.78 |
| C5b | 5-cont | Continuous sentiment \(f(X)\) | **REFUTED** | `native_step5_sentiment.c` — mean S≈+0.0069; common-mode |
| C5c | 5 | Native same-site residual sufficient | **SUPPORTED** | `/tmp/activation_patch_ext.log` |
| C5d | 5 | Foreign \(A_{L12}\) same-site category | **SUPPORTED** | golden **6/7**, mean S=**+0.563**, t=**4.34**, p≈0.005 |
| C5d′ | 5 | Foil-gap SELF/SHUF audit | corroboration | `/tmp/activation_patch_7pair.log` |
| C6 | 6 | New transferable content \(f(X)\) | **REFUTED** | full 7-pair — mean S=**−0.0096**, t≈**−0.03** |
| C6′ | 6 | Soft-serialize (LMHead→softmax→emb) N=1 pilot | promising, non-generalizing | `/tmp/native_step6_soft_serialize` — dog→animal T=0.3 specific vs control, but see C6″ |
| C6″ | 6 | Soft-serialize, full 7-pair | **REFUTED** | scales C6′ down: mean S=**−0.0096**, t≈**−0.03**, sign conflicts within category |
| C7–C8 | 7–8 | Return path / second computation | NOT PURSUED | blocked by C6 |
| **C5e** | **5** | **Scale C5d: n=7 → n=36, 5 categories, frozen native-screened set** | **SUPPORTED** | `/tmp/step5_scaled_set.txt`, `/tmp/step5_scaled_n36.log` — **35/36**, mean S=**+0.5312**, std=0.2052, t=**15.53** (df=35), p≈1e-16 |
| **C5f-i** | **5** | Adversarial control: corr(S, ‖injected‖) | **REFUTED as explanation** | `/tmp/step5_scaled_D_controls.log` — r=**0.002** |
| **C5f-ii** | **5** | Adversarial control: random vector, matched norm, scored vs real leave-one-out | **REFUTED as explanation** | same log — S_random>0 **22/36** (61%), mean=+0.081, t=0.50 (vs real t=15.53) |
| **C5f-iii** | **5** | Adversarial control: permutation of category labels, 200 trials, same real logits | **REFUTED as explanation** | same log — mean(S>0)=15.6/36 (chance), 0/200 shuffles ≥ real mean S |
| **C5g** | **5** | Generalize C5d beyond member→category: object→typical property (5 members families: cold/hot/sweet/sour/dark; hot/sour got 0 screened members, reported not fixed) | **SUPPORTED** | `/tmp/step5_partB_property.log` — n=9 (frozen before latent run), **8/9**, mean S=**+0.1894**, t=**4.03** (df=8), p≈0.004; random-vector control S_random>0 4/9, mean=−0.061 |
| C5h-0 | 5 | PyTorch re-implementation of C5e protocol, GPT-2→GPT-2, new frozen set (calibration) | **VALID instrument** | `xmodel-operand/result_gpt2.json` — 42/44, mean S=+0.522, t=14.97; random t=−0.51; perm 0/200 |
| C5h-1 | 5 | Translator-free operand, GPT-2→DistilGPT2 (shared lineage) | **SUPPORTED** | `xmodel-operand/result_distilgpt2.json` — 45/45, mean S=+0.599, t=20.80; random t=−0.96; perm 0/200 |
| C5h-2 | 5 | Translator-free operand, GPT-2→Pythia-160m (different family) | **INCONCLUSIVE** (preregistered criteria) | `xmodel-operand/result_EleutherAI_pythia-160m.json` — 26/42, mean S=+0.034 (0.7% of native), t=2.04; perm 11/200 |
| C5h-3 | 5 | "Translator-free matters" vs ridge alignment fit on 2000 disjoint words | **NOT SUPPORTED** | `xmodel-operand/RESULT.md` — aligned recovers 89–100% of native S on all three receivers, 9–130× raw; aligned map mostly decodes word identity, so sending the word is a trivial ceiling baseline on this task |

## Architecture split

| Protocol | Ladder role | Functional \(f(X)\)? |
| --- | --- | --- |
| Leaky+tanh \(z\) | C1–C4 | not the Step 5 success path |
| Continuous \(\alpha h\) | C5a, C5b | fail / inconclusive |
| Same-site residual replace | C5c, C5d | success (category) |
| Post-use “is it \(f(X)\)?” | C6 | null |

## Policy

Do not retune α/β/λ/prompts to revive C6. Adversarial-verification pass (C5e/C5f/C5g) targeted the three "still thin" gaps against C5d and did not break it — scale (n=36), confound controls (norm/random/permutation), and a second, structurally different \(f\) (object→property) are now ledgered. Remaining before submission: **site map** (does functional transfer localize to a few \((L,P)\), or work broadly?) — not yet run, no ledger entry. Do not run it adaptively; pre-register the grid before touching results. Fold artifacts into `papers/kv-handoff/evidence.json` before submission.
