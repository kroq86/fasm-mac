# Cross-model operand spike — result (criteria in PREREG.md, unchanged)

Env: ~/ticket_harness/.venv (torch 2.14.0, transformers 5.17.0), CPU, fp32. Commands:
  HF_HUB_OFFLINE=1 python spike.py gpt2 | distilgpt2 ;  python spike.py EleutherAI/pythia-160m
Raw per-word results: result_*.json

| receiver | n | native S | raw S (no translator) | raw S>0 | raw t | random t | perm >=real /200 | aligned S |
|---|---|---|---|---|---|---|---|---|
| GPT-2 (calibration) | 44 | +5.668 | +0.522 | 42/44 | 14.97 | -0.51 | 0 | +5.687 |
| DistilGPT2 | 45 | +5.663 | +0.599 | 45/45 | 20.80 | -0.96 | 0 | +5.533 |
| Pythia-160m | 42 | +5.033 | +0.034 | 26/42 | 2.04 | -1.53 | 11 | +4.455 |

Verdicts:
- Instrument VALID (R0 reproduces paper: +0.522/t=14.97 vs +0.531/t=15.53 on a new frozen set).
- X1 GPT-2 -> DistilGPT2: SUPPORTED (shared lineage; DistilGPT2 is distilled from GPT-2).
- X1 GPT-2 -> Pythia-160m (different family): INCONCLUSIVE by preregistered criteria
  (62% positive, t=2.04; neither >=80%/t>3 nor t<2/<=60%); effect is 0.7% of native.
- "Translator-free matters": NOT SUPPORTED on any receiver. A ridge map fit on 2000
  disjoint words recovers 89-100% of the native effect, 9-130x the raw effect.
Caveat: the aligned map targets the receiver's own embedding of the same word, so its success
mostly shows that the isolated word's identity is linearly decodable from the sender state;
on this task, sending the word itself is a trivial baseline at ceiling.
Original Section 8 logs (/tmp/step5_scaled_*) are lost; this spike used a new candidate set.
