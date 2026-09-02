# GPT-2 124M performance benchmark

Two separate experiments. See PROVENANCE.txt for exact fingerprints/flags/machine.

## 1. Algorithm experiment (VALID): fasm-mac full-recompute vs fasm-mac KV-cache

Same binary, same ISA (x86_64/Rosetta), same real GPT-2 124M weights. Measures
the marginal cost of producing one token's logits at context lengths 4/16/32/64:
prefill a fixed dummy history to length L-1, then time one decode step to L.
Warmup=3 discarded, reps=10 measured, mode order alternated every repetition.
Every single sample's token AND logits max-abs-diff are recorded and checked
against a 1e-3 bound (matching `tensor_gpt2_block0_differential_check.c`'s own
tolerance) before being kept -- a mismatch aborts the run.

Correctness here is a **self-consistency check** between this project's own
two paths, not an independent oracle run inside this program. Both paths were
already independently anchored to real PyTorch by separate, earlier gates
(`tensor_gpt2_full_differential_check.c`, `tensor_gpt2_cached_generation_differential_check.c`)
-- that prior anchoring is what makes self-consistency an adequate bar for a
*performance* harness specifically.

Run: `bench_algorithm.c` -> `raw_algorithm.tsv` (all samples, warmup included
and flagged, per-sample `logits_max_abs_diff`) -> `summarize.py` ->
`summary_algorithm.txt` (median/min/max).

Result (most recent run; numbers vary run to run on a shared dev machine, the
qualitative pattern is what's load-bearing):

| ctxlen | cached median | full median | ratio |
|---|---|---|---|
| 4 | 164ms | 497ms | 3.0x |
| 16 | 297ms | 2004ms | 6.8x |
| 32 | 327ms | 3961ms | 12.1x |
| 64 | 280ms | 7457ms | 26.6x |

`logits_max_abs_diff` was exactly 0 (bit-exact) across every sample in this
run.

**Complexity, precisely (corrected after review):** the cached decode step is
**O(L)**, not O(1) -- the new token's query still attends over all L cached
K/V pairs; only the surrounding per-token projections are O(1) in L. Full
recompute is **O(L^2)** (each of L positions attends over up to L others) plus
O(L) projections. The measured cached times (164/297/327/280ms) are visibly
NOT flat across L=4..64, consistent with O(L), not O(1). What this experiment
supports: **cached decode grows much more slowly with context than full
recompute (O(L) vs O(L^2))** -- an expected, now-measured algorithmic effect,
not a discovery, and not "constant time".

**Not yet measured here** (this file only isolates marginal per-step latency
at fixed context lengths): mode-attributable memory beyond KV-cache capacity.
The `process_peak_rss_kb` column recorded here is a whole-process high-water
mark (shared by both modes in one process) -- NOT a per-mode comparison; the
one genuinely mode-specific memory number is the KV-cache capacity reported
in the PROVENANCE line (`kv_cache_bytes_total_capacity`, ~9MB), since full
recompute allocates no equivalent persistent buffer.

## 1b. Sustained end-to-end generation (`bench_sustained.c`)

Complements the isolated marginal-step measurement above with what a real
generation actually experiences: one prefill + a continuous decode loop,
same real weights, same self-consistency bar (full generated sequences
compared byte-for-byte; a mismatch aborts before any timing is trusted).

A real bug was found and fixed here: the initial version prefilled all 4
prompt tokens, then re-decoded the last prompt token again -- the canonical
cache append wrote a *duplicate* entry rather than reusing/overwriting the
last row, corrupting the cache from step 0 (logits differed from full
recompute by O(1), not float noise, at every single step; argmax merely
survived by luck for 24 steps before flipping). Root-caused by comparing the
live incremental path against fresh full recompute at every step
(`diag_live_path.c`), not just at the sequence's end. Fixed by prefilling
`prompt_len - 1` tokens and decoding the last one separately, matching the
pattern already used correctly elsewhere in this project. After the fix:
bit-exact (`max_abs=0`) agreement at all 40 steps.

Result (prompt_len=4, gen_len=40, single run -- no warmup/repeats yet, unlike
the algorithm experiment above; treat as a first real number, not a matched
benchmark):

| mode | ttft_ms | decode_tokens | decode_s | tokens/sec |
|---|---|---|---|---|
| cached | 590.0 | 39 | 6.03 | 6.47 |
| full | 508.0 | 39 | 117.09 | 0.33 |

~19.4x sustained throughput ratio over a continuous run, consistent in
direction with the algorithm experiment's per-step ratios. TTFT is
comparable between modes here only because the prompt is short (ctxlen=4);
it is not yet measured at longer prompts. This is a single sample on a
shared dev machine -- repeating with warmup/alternation like the algorithm
experiment would strengthen it, not yet done.

## 1c. Per-op profiler (`bench_profile_ops.c`)

Engineering measurement, not a novelty/scientific claim: where does wall
time go, in this project's own scalar-C + canonical-executor path, across
LayerNorm / QKV projection / attention / output projection / FFN expansion
(fc matmul+GELU) / FFN projection / LM head. Baseline: the classical
"transformers are matmul-bound, LM head is disproportionately expensive"
pattern (llm.c, PyTorch-profiler traces of GPT-2-scale models). A real bug
was found and fixed while building this: the LM head's output buffer was
write-only (never read), so `-O2` dead-code-eliminated the entire 768x50257
matmul, reporting a false `0.000ms`. Fixed with an explicit sink read after
every timed phase. Raw output: `profile_ops.txt`.

Result (single instrumented run, no warmup/repeats -- exact numbers from
`profile_ops.txt` on disk):

| phase | prefill L=4 | prefill L=16 | prefill L=32 | prefill L=64 | cached decode (ctx=32) |
|---|---:|---:|---:|---:|---:|
| ffn_expand | 33.8% | 50.9% | 50.8% | 49.2% | 35.2% |
| ffn_proj | 22.9% | 22.8% | 24.7% | 25.4% | 20.4% |
| qkv_proj | 11.6% | 17.0% | 17.5% | 17.9% | 14.8% |
| lm_head | 27.8% | 4.1% | 1.1% | 1.3% | 24.5% |
| out_proj | 3.8% | 5.1% | 5.7% | 5.9% | 4.7% |
| attention | 0.0% | 0.1% | 0.1% | 0.3% | 0.3% |
| layernorm | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

One instrumented run shows MatMul categories account for over 99% of
recorded time. This justifies an Accelerate SGEMM spike, but is not yet a
stable performance profile -- no warmup, no repeats. lm_head at prefill
L=4 (27.8%) is a real, large share -- comparable in weight to FFN there --
not a rounding artifact; it falls off fast as L grows because LM head cost
is roughly fixed per call while FFN cost grows with L.

**Falsifier check**: projections+FFN+LM head are >99% of time at every
point measured; LayerNorm and attention (including the canonical
executor's per-call `ExecStep` dispatch, inside the "attention" column for
cached decode) are consistently <1%. The classical matmul-bound pattern
holds -- not falsified. Executor dispatch overhead is not a bottleneck at
this scale.

**Spike priority** (matmul target, in order): (1) FFN expand + FFN
projection together, 55-75% of time depending on phase; (2) QKV
projection; (3) LM head -- a secondary target for prefill at longer
context, but ~25% of a single *cached decode step*, making it directly
relevant to sustained per-token decode throughput specifically (**not**
because decode is O(1) -- cached attention is still O(L), only the
per-token projections are O(1) in context length; corrected here after
repeating this error once already this session). Padded LM-head vocab
(50257->50304/50432) stays untested until after the Accelerate spike --
its rationale is SIMD/blocking alignment, which the current scalar path
cannot exercise at all.

## 1d. Accelerate SGEMM spike (`bench_accelerate_sgemm.c`)

Scope: does swapping this project's scalar `gpt2_matmul_bias` for Apple's
Accelerate `cblas_sgemm` change wall time, on the actual matmul shapes GPT-2
124M uses, using real layer-0 weights and shape-compatible real-valued
inputs, in one binary. **Makes no claim about Metal, MPS, a planner, or the
model's overall performance** -- narrow backend-swap-cost measurement only.

Build: `clang -arch x86_64 -std=c11 -O2 -Wall -Wextra -Werror
-DACCELERATE_NEW_LAPACK -DGPT2_MAXT=128 bench_accelerate_sgemm.c -framework
Accelerate -lm`. Machine: Apple clang 17.0.0 (clang-1700.6.3.2), macOS
26.2 (25C56), Accelerate framework as shipped with this SDK (no separate
version number exposed by the framework itself). `-DACCELERATE_NEW_LAPACK`
required under `-Werror` (the legacy `cblas_sgemm` header path is
deprecated and would otherwise fail the build).

Method: warmup=3 discarded, 10 measured repetitions per (category, n),
scalar/sgemm order alternated every repetition, every rep's full output
checked for NaN/Inf (a plain diff-vs-bound check would silently PASS a
NaN) and against a 1e-3 bound (tightened from an initial, too-loose 5e-2)
before being kept -- a violation aborts. All 10 raw per-rep timings kept,
not just the median: `raw_accelerate_sgemm.tsv`. Summary: `accelerate_sgemm.txt`.

**Input honesty**: `ln1` (LayerNorm(embedding), real layer-0 weights, a
fixed synthetic token sequence) is the mathematically correct input only
for `qkv_proj`. For `out_proj` (real input: attention output) and
`ffn_expand` (real input: `ln2`, post-attention-residual LayerNorm) it is
a **shape-compatible proxy** -- real dimensions and a real LayerNorm-output
distribution, not the actual per-operation tensor a real forward pass
would produce. `ffn_proj`'s input (`fc_act`) IS the real, correctly-chained
`GELU(fc(ln1))`. This affects the specific correctness-diff values, not
the timing comparison, which is data-independent by construction for BLAS
GEMM.

Result (one process run; numbers vary run to run, qualitative pattern is
load-bearing; independently reproduced separately with matching results:
QKV 22-58x, out_proj 16-42x, ffn_expand 36-122x, ffn_proj 25-64x, lm_head
6x, max diff 2.59e-4):

| category | n | scalar median ms | sgemm median ms | speedup | max_abs_diff |
|---|---:|---:|---:|---:|---:|
| qkv_proj | 4..64 | 6.3-101.8 | 0.25-1.59 | 26-64x | ~6-9e-6 |
| out_proj | 4..64 | 2.1-33.4 | 0.12-0.64 | 18-52x | ~2.7-3.4e-5 |
| ffn_expand | 4..64 | 19.7-298.7 | 0.46-2.22 | 43-134x | ~1.7-2.3e-5 |
| ffn_proj | 4..64 | 9.1-144.0 | 0.40-2.32 | 23-62x | 2.6e-4 |
| lm_head | 1 | 34.9 | 5.5 | 6.3x | 7.6e-5 |

All output diffs are small, well under the tightened 1e-3 bound, and
consistent with float32 summation-order differences between the naive
scalar loop and Accelerate's blocked/vectorized SGEMM -- not a correctness
issue; no NaN/Inf observed. Speedup grows with n for the projection
categories (more amortization of fixed BLAS call overhead), and is much
smaller for `lm_head` (n is always 1 here -- a single row can't benefit
from SGEMM's blocking the way a wider matrix can). This is a
**TRADEOFF-class engineering result, independently verified**: real,
large speedup on the isolated matmul kernel, on this machine, for these
shapes. It does NOT yet establish end-to-end model speedup or logits/
generated-token correctness through the real pipeline -- that is the next
step (below), not yet done.

## 1e. Accelerate SGEMM wired into the real generation path (`bench_accelerate_e2e.c`)

Not an isolated matmul-shape microbenchmark: this runs the SAME real
prefill (`prompt_len-1`) + cached-decode-loop pattern `tensorctl gpt2` and
`bench_sustained.c` use, real GPT-2 124M weights, twice -- once with every
`gpt2_matmul_bias` call site replaced by the Accelerate-backed equivalent,
once scalar/unmodified. The canonical `CAUSAL_ATTENTION_CACHED` executor
is identical in both runs; only the surrounding projection/FFN/LM-head
matmuls swap backend. Does NOT modify `fasm/examples/tensorctl_gpt2.c` or
`tensor_gpt2_forward.h` -- a standalone comparison harness in scratchpad,
reusing the identical real weights and prefill/decode pattern those files
use, to avoid touching shared/committed production files without a
separate, explicit integration decision.

Full generated token sequences (44 tokens: prompt_len=4 + gen_len=40) are
compared byte-for-byte before any timing is trusted -- a mismatch would
abort with no numbers printed. Result across 3 repeated process runs (not
a formal warmup+10-rep+alternating harness like `bench_sustained.c` --
each number below is one full process run, repeated 3 times to check
reproducibility, not to compute a statistically rigorous median):

| run | scalar ttft_ms | scalar tok/s | sgemm ttft_ms | sgemm tok/s | ttft speedup | tok/s speedup |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 586.4 | 4.31 | 58.9 | 42.09 | 9.96x | 9.75x |
| 2 | 680.9 | 5.96 | 53.7 | 41.56 | 12.68x | 6.98x |
| 3 | 649.0 | 6.29 | 51.8 | 44.44 | 12.52x | 7.06x |
| 4 (saved, `accelerate_e2e.txt`) | 566.3 | 6.13 | 64.0 | 45.29 | 8.84x | 7.38x |

Sequences were byte-identical in all 4 runs. Sustained decode throughput
speedup is consistently **7-10x** (noisier and generally lower than the
isolated-kernel 20-136x from §1d -- expected, since a real decode step
spends real time in LayerNorm, the canonical executor's attention/cache
read, and small per-layer glue that SGEMM doesn't touch, so Amdahl's law
caps the achievable end-to-end speedup well below the isolated matmul
speedup). **Verdict: SUPPORTED** for "Accelerate makes the real generation
path faster, with byte-identical output" -- this is a single machine,
single process type, not yet a formal repeated-warmup benchmark; the
qualitative direction (large, consistent speedup, correctness preserved)
is the load-bearing claim, not the exact 7x vs 10x figure.

**Next step, not yet done**: formalize this into a warmup+10-rep+
alternating-order harness matching `bench_sustained.c`'s rigor, and decide
whether to actually wire this into `tensor_gpt2_forward.h`/
`tensorctl_gpt2.c` as a real backend option (a separate integration
decision -- this harness proves it's worth doing, it doesn't do the
integration itself). Padded-vocab (50257->50304/50432) and an SGEMV
variant for the `n=1` LM head case stay deferred -- `lm_head`'s isolated
speedup (6.3x, weakest of the five categories in §1d) doesn't establish
that vocab size limits Accelerate; testing padding now would be
premature.

## 2. Implementation experiment (fasm-mac vs. llm.c): BLOCKED

`bench_gpt2.c` (in `../llmc_baseline/llm.c/`, a minimal driver built on
llm.c's own unmodified `gpt2_forward`/`gpt2_build_from_checkpoint`) exists
and produces numbers, but they must NOT be compared against fasm-mac's as an
implementation-performance result. See PROVENANCE.txt's
CROSS_RUNTIME_INCONCLUSIVE note: fasm-mac runs x86_64-under-Rosetta on this
machine, llm.c runs native arm64 -- an ISA mismatch no amount of careful
timing methodology corrects for.

Two possible unblock paths, not yet attempted:
- build a native arm64 implementation of this project's canonical executor
  ABI (`ExecStep` contract) -- a separate, not-yet-scoped port; or
- build llm.c itself under x86_64/Rosetta (or run both implementations on a
  real x86_64 machine) so both sides share fasm-mac's current ISA instead of
  porting fasm-mac. Likely the smaller lift of the two, not yet tried.

Earlier single-run numbers from both `bench_gpt2.c` and `bench_fasmmac.c`
(no warmup, no repeats, mixed ISA) were reported informally before this
methodology existed and must be treated as superseded, not as evidence.
