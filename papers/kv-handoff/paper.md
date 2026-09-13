# From Transferred Cache to Usable Memory: A Small-Model Case Study with Native KV-Handoff Reproduction

## Abstract

Transferring a language model's key–value cache can change another model's predictions without establishing that the receiver can use the transferred information. We report a retrospective case study of frozen GPT-2 and DistilGPT2 connected by task-specific rank-8 residual KV adapters with 147,456 trainable parameters. An order-dependent binary classification checkpoint achieves 31/32 correct predictions on its original development set; an opposite-class cache replacement reduces accuracy to 1/32. A separately trained five-row lookup adapter reports 20/32 exact two-token answers on a held-out test split, compared with 1/32 under document replacement and 1/32 under binding permutation. Conversely, an abstract two-key task has zero top-1 accuracy despite a positive likelihood contrast, and a recombined XOR task reaches only 14/32 correct predictions. These experiments do not isolate the cause of the between-task differences. We additionally reproduce the saved ordering checkpoint in an independent native execution path: selected sender caches, adapted caches, and receiver logits agree with a CPU reference within a fixed numerical tolerance on all 32 cases, including the same classification error. The report contributes a bounded, inspectable reproduction and an artifact-linked account of successes and limitations, not a new cache-transfer method, evidence of general reasoning, or a demonstrated efficiency advantage.

## 1. Introduction

The engineering question is straightforward: can a sender compute a representation once and provide it directly to a different receiver, without generating an intermediate textual message? The scientific question is harder: what evidence establishes that the receiver uses the *particular information* in that representation?

Three claims must remain distinct. First, a cache adapter may be executable. Second, its output may change the receiver's likelihoods or answers. Third, those changes may enable correct use of sender-private information. Neither of the first two claims entails the third. Moreover, successful retrieval of one association does not establish compositional use of multiple associations.

The central thesis is that a distributional effect, correct use of transferred information, and compositional use of that information are distinct evaluation criteria. In this closely related model pair, an observed likelihood effect can coexist with zero correct answers, while successful retrieval in one experiment does not establish compositional use in another. The experiments do not identify a universal ordering of task difficulty or isolate the cause of their differences. The native reproduction makes one positive checkpoint inspectable; it is supporting implementation evidence, not a second claim about model capability or a new test population.

The study is retrospective. Experiments were developed sequentially, with changes to datasets and optimization settings during the research process. We preserve their individual evaluation boundaries rather than describe the entire sequence as one preregistered benchmark. No new training was performed to prepare this report.

## 2. Related work and scope of contribution

Cache-to-Cache already learns projections and fusion between language-model KV caches. Accordingly, learned cross-model cache communication is not our novelty claim. [Fu et al., Cache-to-Cache](https://arxiv.org/abs/2510.03215).

Latent Cache Flow studies smaller cache translators and communication across different contexts. Its scope also prevents treating a compact adapter or sender-private context alone as a new contribution. [Rossi et al., Latent Cache Flow](https://arxiv.org/abs/2605.22863).

Closed-form linear transfer for prefill reuse and heterogeneous dual-cache communication further establish that linear mappings and cross-model interfaces have existing precedents. We do not run matched comparisons against these methods. [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893), [Dual-Cache Latent Space Communication](https://arxiv.org/abs/2608.20617).

Most directly, Cheng et al. audit whether cache effects depend on the identity of the relayed example. Their mismatched, zeroed, and moment-matched controls distinguish private-information transfer from generic effects across substantially broader settings. Our content-specificity question therefore also has a direct measurement precedent. Our narrower contribution is the particular small-model task record and native checkpoint reproduction, not the invention of causal cache auditing. [Cheng et al., When Does Latent Communication Pay?](https://arxiv.org/abs/2608.04893).

## 3. Model and interface

The sender is GPT-2 124M with twelve transformer blocks. The receiver is DistilGPT2 with six blocks. Both use hidden width 768 and twelve attention heads of dimension 64. Both base models remain frozen. This is a favorable, closely related pair; results do not establish compatibility across unrelated model families, tokenizers, positional encodings, or attention geometries.

For receiver layer l, we select sender layers [1, 3, 5, 7, 9, 11], using zero-based indices. Each selected cache stream is flattened across heads into X of shape T × 768. Keys and values have separate residual low-rank maps at each receiver layer:

```text
X' = X + (X A) B
A: 768 × 8; B: 8 × 768
parameters = 6 layers × 2 streams × (768×8 + 8×768) = 147,456
```

The map can mix head coordinates; it is not merely independent rescaling within each head. Low rank describes the correction, not the total rank of the identity-plus-correction transform. Different tasks train different adapters. The experiment collection is not evidence for a single universal bridge.

The receiver starts with the adapted prefix cache and processes its own query at positions following that prefix. It never receives the sender's prefix as receiver input text. It subsequently uses ordinary causal attention over the imported memory and its own query states.

## 4. Evaluation and controls

We distinguish task accuracy, content-specific likelihood effects, and implementation agreement. For lookup, per-document NLL is the mean negative log probability across the two target tokens. The reported exact score requires both teacher-forced next-token argmax predictions to match. Earlier target tokens are supplied when scoring later ones; the result is not a separately executed free-running generation benchmark. Causal shifting prevents the token being scored from being its own input at that prediction position.

This distinction should not understate the exact-match metric: for a deterministic causal decoder with identical numerics and tie-breaking, if every teacher-forced argmax matches the target, greedy decoding follows the same target prefix by induction. Conversely, a first greedy error makes exact match fail. Thus the all-token criterion corresponds to greedy exact match over the fixed target length under those assumptions, unlike mean token accuracy or NLL. It does not verify a separate cached-generation implementation, stopping behavior, or continuation beyond the target; those were not run for lookup.

The document-level paired contrast is NLL(control) − NLL(correct). A positive contrast favors correctly paired memory. The lookup script resamples these paired document differences 10,000 times and reports empirical 95% endpoints. These intervals concern NLL, not accuracy differences. They condition on one fitted adapter and one small split; they do not measure training-seed uncertainty. Multiple exploratory comparisons were performed without a study-wide multiplicity correction.

The controls answer different questions:

- **Different-document cache:** does pairing matter, beyond the presence of a cache?
- **Zero cache:** does the channel matter, while acknowledging a possible distribution shift?
- **Wrong binding:** does reassignment of values to names change performance when names and values themselves remain present?
- **Changed query:** can the same table cache support another requested association?
- **Position shuffle:** does destroying sequence order before sender encoding impair an order-dependent target?

An exact classical parser/counting procedure given the original input is the appropriate functional baseline for these synthetic tasks. No superior capability over that baseline is claimed. Raw-context transmission and receiver re-prefill are also essential efficiency baselines, but have not been measured here. Degraded-cache controls are mechanism tests, not substitutes for those competitors.

## 5. Results

The table separates experimental units and evaluation roles. Source labels refer to entries in `evidence.json`; each entry records the original path and hash. Distinct adapters must not be treated as one jointly capable system.

| Experiment / adapter | Evaluation population | Main measurement | Relevant control | Reproduction status | Supported interpretation |
| --- | --- | --- | --- | --- | --- |
| Ordering / saved ordering adapter | 32 original dev sequences | 31/32 class predictions | Opposite-class cache 1/32; shuffled input positions 17/32 | Saved weights; inference replay and native differential | Task information and input order affect this selected checkpoint |
| Lookup / separately trained lookup adapter | 32 held-out test documents | 20/32 exact two-token targets | Wrong binding 1/32; changed query 23/32 | Stored source/result only; fitted weights not saved | Name-specific use in this setting; not same-checkpoint replay |
| Abstract two-key / separately trained two-key adapter | Original dev evaluation | Zero top-1; positive pairing NLL contrast | Wrong-key contrast includes zero | Stored source/result; not replayed for this report | Likelihood effect without demonstrated usable recall |
| XOR / separately trained XOR adapter | 32 original dev inputs and 32 recombined inputs | 22/32 original; 14/32 recombined | Half-swap interval includes chance | Stored source/result; not replayed for this report | No supported recombination generalization in this regime |
| Native reproduction / same saved ordering adapter | Same 32 original dev sequences | 32/32 predictions agree with oracle | All selected/adapted caches and logits checked | Runnable native artifact | Implementation fidelity, not additional independent accuracy evidence |

### 5.1 Order-dependent classification and checkpoint replay

The sender sees 32 A/B symbols, always sixteen of each. Class depends on which symbol is the majority in the first sixteen positions; the second half has the reverse majority. Total symbol counts cannot solve the task, while a simple first-half count can. The receiver sees only `Class:` after cache import. The historical filename calls this “mini-GunPoint,” but it is a synthetic sequence task, not the GunPoint dataset.

The saved adapter obtains the following original-development results:

| Condition | Correct / 32 |
| --- | ---: |
| Correct cache | 31 |
| Different-document, opposite-class cache | 1 |
| Input position shuffle before sender encoding | 17 |
| Correct keys, wrong-document values | 4 |
| Wrong-document keys, correct values | 28 |
| Query after 32 fixed filler tokens | 31 |

The different-document condition uses a one-row roll of alternating class labels, deliberately yielding an opposite-class replacement. Its 1/32 result must not be described as an unbiased random-cache baseline. The pattern supports sensitivity to transferred task information and order, within this selected checkpoint and development set.

K/V swaps show a larger degradation from wrong values in this particular experiment. They do not establish that keys are generally unimportant, nor directly measure query–key geometric alignment. The tested reduced-layer configurations achieve 16/32, whereas keeping the first half of heads retains 31/32. These interventions characterize this trained adapter; they do not prove every individual layer is indispensable or that a retrained reduced interface would fail.

The 32-token delay preserves accuracy but is a fixed-filler proxy, not autonomous long-horizon generation. No retention curve or interference threshold follows. A fresh inference-only replay reproduces all 22 historical accuracy fields; floating-point summaries differ slightly. The replay is not bit-identical to the historical MPS result.

Additional local sensitivity checks on that same development set are retained rather than promoted to general robustness claims:

| Intervention on the saved ordering adapter | Result |
| --- | --- |
| Scale both adapted K and V by 0.5 / 1 / 2 | 31/32 / 31/32 / 30/32 correct |
| Add Gaussian noise with standard deviation 0.1 times each cache tensor's standard deviation | 31/32 correct for the tested draw |
| Use the tested weak conflicting query | 31/32 correct; prediction flip rate 0 |

These are a few fixed interventions, not a sweep over failure thresholds, noise seeds, or adversarial prompts. Unchanged accuracy is not proof of equivalence; in particular, scaling by two changes one classification.

### 5.2 Five-row lookup: a positive but separately trained result

The lookup sender receives a five-row name-to-ID table with shuffled row order. Names in the actual artifact are Bob, Alice, Ivan, Lena, and Anna. IDs are selected to have two target tokens. The receiver is asked for one name's ID. The script uses 64 training, 32 development, and 32 test documents; it selects the adapter on development loss before evaluating test. The stored run uses rank 8, Adam learning rate 0.01, seed 20260904, and stops after 80 epochs.

| Test condition | Exact / 32 | Mean target-token NLL |
| --- | ---: | ---: |
| Correct cache | 20 | 1.447434 |
| Different-document cache | 1 | 15.385736 |
| Wrong binding | 1 | 6.776008 |
| Zero sender cache | 0 | 10.462410 |
| Different query name, same table (secondary) | 23 | 1.605134 |

Wrong binding rotates the IDs across names while retaining row order. The correct-versus-control paired NLL differences are 13.938302 for document replacement, 5.328573 for binding permutation, and 9.014977 for zero memory. Their respective 95% intervals are [11.998692, 15.731031], [3.913476, 6.778359], and [8.195748, 9.724291]. These are observed, unadjusted intervals from the stored run.

Together, these results support usable name-specific information in that controlled setting. The secondary query condition tests reuse of a table with another familiar name, not unseen query-template generalization. Critically, this script saved results but not its fitted adapter. We source-reviewed its scoring and report its stored measurements; we did not replay its trained weights or integrate lookup into the native executable.

### 5.3 Where the evidence stops

An abstract two-key experiment reports zero top-1 accuracy even with correct memory. Its correct-cache NLL is 6.076248, versus 6.263900 with a different document. The small pairing contrast has interval [0.122524, 0.256115], but the wrong-key contrast spans zero, [−0.315573, 0.311536]. Thus a statistically detectable likelihood effect does not establish usable retrieval. The failure also does not isolate a specific addressing defect because correct-condition recall itself fails.

A separately trained XOR composition experiment reports 22/32 on its original development inputs and 14/32 on recombined halves scored against the resulting XOR label. Its half-swap accuracy interval is [0.28125, 0.625], including chance. The stored run uses 800 epochs and development-accuracy selection; the best recorded checkpoint is at epoch 180. We report no supported compositional generalization in this regime, not an architectural impossibility.

The half-swap intervention splices halves of the original symbol sequences and re-encodes the complete new sequence with the sender. It does not concatenate independently computed KV caches; cache composability is untested. For XOR, the separate position-shuffle NLL contrast has interval [−0.464881, 3.600663], crossing zero. The clearer ordering-task shuffle result therefore cannot be attributed to XOR as well. Neither shuffle establishes exact preservation of positional representations.

Lookup success and abstract two-key failure are not contradictory measurements of an identical condition. Their tasks and training procedures differ. The available record does not isolate which difference causes the outcome. Similarly, changes to optimization regimes elsewhere in the project cannot retroactively invalidate or repair these experiments without new controlled comparisons.

## 6. Native reproduction of the saved ordering checkpoint

The runnable artifact uses the project's C primitives, Accelerate matrix multiplication, self-written tokenizer and safetensors reader, and canonical FASM cached-attention executor. An additive layer-count loader supports the six-layer receiver while preserving the original twelve-layer entry point. The executable does not invoke PyTorch. An external CPU/eager PyTorch environment prepares reference tensors and losslessly exports the receiver's original weight container to safetensors; both models and the bridge are SHA-pinned.

For each of the same 32 development sequences we compare selected sender K/V, adapted K/V before receiver query processing, and all 50,257 output logits. This covers 768 cache tensors and 32 logit tensors. The fixed acceptance rule is:

```text
abs(native − reference) <= 0.001 + 0.0001 * abs(reference)
```

Non-finite values fail. All cases pass; the largest absolute difference is 0.00103759766, allowed by the combined bound. All 32 predicted tokens match, including the reference's one classification error. This distinguishes successful software reproduction from an improvement in model accuracy.

The import copies into receiver-owned storage, leaves the sender intact, requires an empty receiver, and commits cache positions only after validating the transformed cache. Tests cover occupied import, inconsistent/full/empty positions, non-finite inputs, overflow rollback, ownership and malformed artifact metadata. Two identical native invocations produce identical output. Existing GPT-2 CLI and cached/full differential gates also pass. The handoff gate additionally passed from an exported copy of the exact staged Git tree before commit 1453cbd2, with the external bundle supplied explicitly.

This is a bounded standalone executable, not a general cache ABI or a new `tensorctl` subcommand. Its capacity is 64 positions. The demonstrated 32-symbol prompt occupies 35 prefix positions; `Class:` extends the receiver to 37. At this receiver geometry, FP32 KV storage is 36,864 bytes per position, or 1,290,240 bytes for that prefix, excluding other buffers. That arithmetic is not a memory benchmark. Avoiding intermediate text generation does not imply lower communication volume than sending token IDs.

## 7. Limitations and interpretation

The principal limitations are the single closely related model pair, tiny synthetic datasets, task-specific adapters, development selection for ordering and XOR, lack of training-seed replication, and missing saved lookup weights. Original result JSON files provide summaries rather than a complete independently regenerated per-example statistical audit. Tokenization and prompt familiarity may affect outcomes. Moment-matched random-cache controls, systematic length degradation, independent template families, and matched end-to-end cost comparisons are absent.

The positive results establish task-conditioned use of transferred information, not an information-theoretic channel capacity. A 32-symbol classification input can yield just one task-relevant bit. Reduced-head tolerance is not a compression benchmark; fixed-filler retention is not durable agent memory. Native verification establishes agreement with the chosen oracle on tested inputs, not universal numerical equivalence or semantic correctness of arbitrary imported memory. Cache shape validity is not evidence that its content is trustworthy.

We therefore separate three conclusions. The ordering bridge is implemented and reproducible. Stored lookup evidence supports a more selective use of memory, but lacks same-checkpoint replay. A broad advantage over text/context transfer, classical algorithms, other cache translators, or more capable models remains unestablished. None of these conclusions requires treating the project as a new reasoning architecture.

## 8. Conclusion

In this GPT-2-to-DistilGPT2 case study, transferred caches can support order-dependent classification and, with a different trained adapter, selective lookup. Those successes coexist with unusable two-key recall and unsupported recombined XOR performance. The causal explanation of the between-task variation is unresolved. A native reconstruction reproduces one saved ordering adapter through intermediate caches and final logits without retraining. The resulting artifact provides a concrete basis for checking one learned channel; it does not collapse mechanism, generalization, and efficiency into a single success claim.

## Appendix A. Evidence map and boundaries

This map replaces a single sequence of checkmarks such as “capacity -> addressing -> composition.” Those labels mix different constructs and experiments. **Observed** denotes a measurement in the specified setting, **not established** denotes an inference the evidence does not support, and **not tested here** denotes a missing controlled measurement in the report's evidence set. It is not a new experimental verdict or a complete audit of every historical spike.

| Question | Evidence and boundary |
| --- | --- |
| Content-specific use | Observed in ordering and stored lookup; abstract two-key shows a smaller likelihood contrast without usable answers. Not a universal result across all adapters. |
| Addressing / binding / selective retrieval | Lookup supports name-specific retrieval under binding and query interventions. Failed two-key recall does not isolate an addressing defect. These are different conditions, not a logical contradiction. |
| Global versus local information | Both a global classification and one local lookup experiment succeed. The cause of between-experiment differences is not isolated. |
| Value compatibility / query–key alignment | Wrong-V replacement hurts ordering more than wrong-K replacement. This is intervention sensitivity, not a measured fraction of information in V or an attention-score alignment analysis. |
| Position / order | Input permutation affects ordering; the XOR position-shuffle NLL interval includes zero. Exact position preservation is not established. |
| Layer participation / partial cache | Tested first-layer, last-layer and first-half-layer configurations underperform the full ordering interface. Not exhaustive leave-one-out tests; no retrained reduced interface. “All six individually necessary” is unsupported. |
| Head participation / redundancy | The tested first-half-head mask retains accuracy. No general head correspondence, rank, or compression conclusion follows. |
| Scale / noise / conflict | Local checks are reported in Section 5.1. No equivalence test, robustness boundary, or strong contradiction-handling result. |
| Persistence | Fixed filler delay preserves development accuracy. Autonomous generation, long-horizon retention curves and interference limits are not established. |
| Composition / cache merging | XOR recombination is unsupported in its tested regime. Ready-made cache concatenation, multiple senders and incremental updates are not tested here. |
| Bandwidth / capacity | Input length and successful labels are not information-theoretic capacity measurements. Neither 32 symbols nor a five-row table implies complete recovery of all input information. |
| Task dependence / evaluation sensitivity | Heterogeneous results are observed, but tasks, optimization and scoring differ. No matched factorial experiment identifies naturalness, semantic complexity or instrument sensitivity as the cause. |
| Generalization | Lookup has a held-out document split and a secondary familiar-name query. Independent ordering test, new template families and training-seed replication remain absent. |
| Optimization / sample or adapter complexity | No matched optimization, data-size or adapter-capacity study is included here. Earlier hidden-state-injection rungs must not be pooled with these KV experiments as one capacity curve. No universal optimization fix is claimed. |
| Geometry / precision | One fixed 12-to-6 layer selection is exercised, with equal head geometry and related tokenizers. Alternative mappings, unequal head widths/counts, RoPE transfer, FP16 dependence and unrelated model families are not tested here. |
| Causal necessity / wrong memory | Chosen cache replacements affect outputs and pairing scores. This does not prove every component necessary or establish calibrated rejection of wrong memory. |
| Efficiency / deployment | Native agreement is tested. Matched latency, peak-memory, communication savings and precision/compression tradeoffs are not measured. |

Single-fact transfer and earlier scalar/per-channel/dense hidden-state experiments belong to the wider exploratory history, not the quantitative evidence set of this manuscript. We do not infer their final status from the supplied narrative inventory or silently fold them into a KV-adapter hierarchy. The source ledger intentionally identifies the experiments supporting this report rather than presenting the entire repository as a single controlled study.

## Appendix B. Artifact record

The accompanying `evidence.py` prints or checks `evidence.json`, extracting values from the repository's original result files and recording SHA-256 hashes. This checks manuscript-source consistency; it does not rerun training or recompute confidence intervals. `README.md` gives the native reproduction command and submission-readiness limitations. Main native source revision: [1453cbd2](https://github.com/kroq86/fasm-mac/commit/1453cbd2).

Author identity, affiliations, final author review, and a submission-formatted export remain to be completed. This document is a research draft, not an arXiv submission or a claim of acceptance.
