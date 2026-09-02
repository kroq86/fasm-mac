/* Preregistered ALGORITHM experiment: fasm-mac full-recompute vs
 * fasm-mac canonical-KV-cache decode, same runtime, same ISA (x86_64
 * under Rosetta on this machine -- both modes run through the identical
 * binary/build, so ISA is matched by construction here). This is NOT a
 * cross-runtime (vs llm.c) comparison -- that comparison is currently
 * BLOCKED: this project's canonical executor is x86_64 FASM only, no
 * ARM64 executor ABI exists, so a native-ARM64 fasm-mac build does not
 * exist to compare against native-ARM64 llm.c. See
 * scratchpad/gpt2_perf_bench/README.md for the explicit
 * CROSS_RUNTIME_INCONCLUSIVE verdict and what would unblock it.
 *
 * Per-context-length marginal cost: for each context length L in
 * {4,16,32,64}, prefill a fixed L-1 token history, then measure ONE
 * decode step transitioning to context L (cached: one real
 * compile()-once/execute-many attention step per layer plus the
 * surrounding plain-C ops; full: one full recompute over all L
 * positions).
 *
 * Complexity, precisely (corrected after review -- an earlier version of
 * this file's comment and STATUS.md both wrongly called the cached path
 * O(1)): the new token's query still attends over every one of the L
 * cached K/V pairs, so ONE cached decode step is O(L), not O(1); the
 * surrounding per-token MATMUL/BIAS_ADD/GELU/LM-head work is O(1) in L
 * (only the one new token is processed), but the attention term is not.
 * A full-recompute step is O(L^2) for its attention (each of the L
 * positions attends over up to L others) plus O(L) for its projections,
 * so overall O(L^2). The measured cached times (154/172/269/276 ms
 * across L=4/16/32/64, NOT flat) are consistent with this -- a true O(1)
 * path would show ~constant latency, which is not what was measured.
 * What this experiment supports is: cached decode grows much more slowly
 * with context than full recompute (O(L) vs O(L^2)), not "constant time".
 *
 * Correctness here is a SELF-CONSISTENCY check between this project's own
 * two paths (cached vs full), not an independent external oracle run
 * inside this program. Both paths were already independently anchored
 * against real PyTorch by separate, earlier gates
 * (tensor_gpt2_full_differential_check.c,
 * tensor_gpt2_cached_generation_differential_check.c) -- that is what
 * makes self-consistency an adequate correctness bar for a performance
 * harness specifically, not a claim that this file re-establishes
 * independent correctness on its own.
 *
 * Every measured step's predicted next token AND full logits vector
 * (max abs diff recorded per row, tolerance 1e-3 -- matching
 * tensor_gpt2_block0_differential_check.c's bound, not a looser one) is
 * compared between cached and full -- a mismatch aborts the run (exit
 * nonzero) rather than silently reporting timing over an unverified
 * computation. Correctness is re-checked on every single sample, not
 * once at the start.
 *
 * This file measures ONLY isolated marginal per-step latency at fixed
 * context lengths. It does NOT measure TTFT, sustained tokens/sec over a
 * real multi-step generation run, or mode-attributable memory (the
 * recorded process_peak_rss_kb is a whole-process high-water mark, not
 * split between modes -- see the comment at its call site). Those are
 * separate, not-yet-built measurements; this file's result alone does
 * not close roadmap item 5.
 *
 * Warmup + repeated measurement, alternating cached/full order every
 * repetition (and swapping which one goes first every other repetition)
 * to symmetrize thermal/load drift. All raw per-sample timings (warmup
 * included, flagged) are written to a TSV for independent analysis --
 * this program never reports only a pre-reduced median.
 *
 * Fails closed: context length, repetition count, and warmup count are
 * validated before any measurement; an out-of-range value is a hard
 * error, not a silently-clamped one.
 */
#include "tensor_semantic_compiler.h"
#include "tensor_gpt2_forward.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>

#if M != 768 || H != 12 || D != 64 || QW != 2304 || MAXCACHE < 128
#error "requires -DM=768 -DH=12 -DD=64 -DQW=2304 -DMAXCACHE>=128"
#endif
#if GPT2_MAXT < 128
#error "requires -DGPT2_MAXT>=128 (tensor_gpt2_forward.h)"
#endif

typedef struct {
    Node g[3];
    ExecStep steps[8];
    Context ctx[8];
    uint32_t count;
    float x[768 * 3], gx[768 * 3];
    float cache_data[MAXCACHE * 768], cache_grad[MAXCACHE * 768], cache_aux[MAXCACHE * 768];
    float out[768], gout[768];
} LayerAttnGraph;
static LayerAttnGraph g_lag[GPT2_NLAYER];
enum { LAG_X, LAG_CACHE, LAG_ATT, LAG_NODES };

static int setup_layer_attn_graph(LayerAttnGraph *lag) {
    memset(lag, 0, sizeof *lag);
    lag->g[LAG_X] = (Node){LEAF, NONE, NONE, INPUT, {lag->x, lag->gx, NULL, 1, QW, 0}};
    lag->g[LAG_CACHE] = (Node){LEAF, NONE, NONE, INPUT, {lag->cache_data, lag->cache_grad, lag->cache_aux, (uint32_t)MAXCACHE, (uint32_t)M, 0}};
    lag->g[LAG_ATT] = (Node){CAUSAL_ATTENTION_CACHED, LAG_X, LAG_CACHE, TEMP, {lag->out, lag->gout, NULL, 1, M, 0}};
    return compile(lag->g, LAG_NODES, .01f, lag->steps, lag->ctx, 8, &lag->count);
}
static void reset_layer_attn_graph(LayerAttnGraph *lag) {
    lag->g[LAG_CACHE].tensor.aux_count = 0;
}

static void gpt2_prefill_cached_upto(const Gpt2Weights *w, const int *token_ids, int upto /* fills cache with upto-1 positions, leaves position upto-1 for the caller to step */, float *out_last_logits) {
    if (upto <= 1) return; /* nothing to prefill; caller steps position 0 directly */
    int n = upto - 1;
    static float h[GPT2_MAXT * GPT2_M];
    for (int t = 0; t < n; t++) for (int m = 0; m < GPT2_M; m++)
        h[t * GPT2_M + m] = w->wte[token_ids[t] * GPT2_M + m] + w->wpe[t * GPT2_M + m];
    static float ln1[GPT2_MAXT * GPT2_M], qkv[GPT2_MAXT * GPT2_QW], attn[GPT2_MAXT * GPT2_M];
    static float proj[GPT2_MAXT * GPT2_M], res1[GPT2_MAXT * GPT2_M], ln2[GPT2_MAXT * GPT2_M];
    static float fc[GPT2_MAXT * GPT2_F], act[GPT2_MAXT * GPT2_F], fcproj[GPT2_MAXT * GPT2_M];
    for (int L = 0; L < GPT2_NLAYER; L++) {
        const Gpt2BlockWeights *bw = &w->blk[L];
        gpt2_layernorm_raw(h, n, GPT2_M, ln1);
        gpt2_matmul_bias(ln1, n, GPT2_M, bw->attn_w, bw->attn_b, GPT2_QW, qkv);
        gpt2_causal_attention(qkv, n, attn);
        for (int t = 0; t < n; t++) {
            memcpy(&g_lag[L].cache_data[t * 768], &qkv[t * GPT2_QW + GPT2_M], sizeof(float) * 768);
            memcpy(&g_lag[L].cache_aux[t * 768], &qkv[t * GPT2_QW + 2 * GPT2_M], sizeof(float) * 768);
        }
        g_lag[L].g[LAG_CACHE].tensor.aux_count = (uint32_t)n;
        gpt2_matmul_bias(attn, n, GPT2_M, bw->projw, bw->projb, GPT2_M, proj);
        for (int i = 0; i < n * GPT2_M; i++) res1[i] = h[i] + proj[i];
        gpt2_layernorm_raw(res1, n, GPT2_M, ln2);
        gpt2_matmul_bias(ln2, n, GPT2_M, bw->fc_w, bw->fc_b, GPT2_F, fc);
        gpt2_gelu(fc, n * GPT2_F, act);
        gpt2_matmul_bias(act, n, GPT2_F, bw->fcproj_w, bw->fcproj_b, GPT2_M, fcproj);
        for (int i = 0; i < n * GPT2_M; i++) h[i] = res1[i] + fcproj[i];
    }
    (void)out_last_logits; /* prefill here is purely to populate the cache; logits at n-1 unused by this harness */
}
static int gpt2_decode_step_cached(const Gpt2Weights *w, int token_id, int pos, float *logits_out) {
    float h[GPT2_M];
    for (int m = 0; m < GPT2_M; m++) h[m] = w->wte[token_id * GPT2_M + m] + w->wpe[pos * GPT2_M + m];
    for (int L = 0; L < GPT2_NLAYER; L++) {
        const Gpt2BlockWeights *bw = &w->blk[L];
        float ln1[GPT2_M]; gpt2_layernorm_raw(h, 1, GPT2_M, ln1);
        float qkv[GPT2_QW]; gpt2_matmul_bias(ln1, 1, GPT2_M, bw->attn_w, bw->attn_b, GPT2_QW, qkv);
        float attn[GPT2_M];
        memcpy(g_lag[L].x, qkv, sizeof(float) * QW);
        if (tensor_transformer_steps_execute(g_lag[L].steps, g_lag[L].count)) return -1;
        memcpy(attn, g_lag[L].out, sizeof(float) * 768);
        float proj[GPT2_M]; gpt2_matmul_bias(attn, 1, GPT2_M, bw->projw, bw->projb, GPT2_M, proj);
        float res1[GPT2_M]; for (int i = 0; i < GPT2_M; i++) res1[i] = h[i] + proj[i];
        float ln2[GPT2_M]; gpt2_layernorm_raw(res1, 1, GPT2_M, ln2);
        float fc[GPT2_F]; gpt2_matmul_bias(ln2, 1, GPT2_M, bw->fc_w, bw->fc_b, GPT2_F, fc);
        float act[GPT2_F]; gpt2_gelu(fc, GPT2_F, act);
        float fcproj[GPT2_M]; gpt2_matmul_bias(act, 1, GPT2_F, bw->fcproj_w, bw->fcproj_b, GPT2_M, fcproj);
        for (int i = 0; i < GPT2_M; i++) h[i] = res1[i] + fcproj[i];
    }
    float lnf[GPT2_M]; gpt2_layernorm_raw(h, 1, GPT2_M, lnf);
    gpt2_matmul_bias(lnf, 1, GPT2_M, w->lnf_w_folded, w->lnf_b_folded, GPT2_VOCAB, logits_out);
    return 0;
}
static int greedy_argmax(const float *logits, int vocab) {
    int best = 0; for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i; return best;
}
static double ns_between(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e9 + (double)(b.tv_nsec - a.tv_nsec);
}
static long peak_rss_kb(void) {
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024; /* macOS reports bytes, not KB */
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s SAFETENSORS_PATH OUT_TSV WARMUP REPS CTXLEN[,CTXLEN...]\n", argv[0]);
        return 2;
    }
    const char *safetensors_path = argv[1];
    const char *out_tsv_path = argv[2];
    int warmup = atoi(argv[3]);
    int reps = atoi(argv[4]);
    if (warmup < 0 || warmup > 100) { fprintf(stderr, "warmup out of bounds [0,100]: %d\n", warmup); return 2; }
    if (reps < 1 || reps > 1000) { fprintf(stderr, "reps out of bounds [1,1000]: %d\n", reps); return 2; }

    int ctxlens[16], n_ctxlens = 0;
    {
        char buf[256]; snprintf(buf, sizeof buf, "%s", argv[5]);
        char *tok = strtok(buf, ",");
        while (tok) {
            if (n_ctxlens >= 16) { fprintf(stderr, "too many context lengths (max 16)\n"); return 2; }
            int v = atoi(tok);
            if (v < 2 || v > GPT2_MAXT) { fprintf(stderr, "context length out of bounds [2,%d]: %d\n", GPT2_MAXT, v); return 2; }
            ctxlens[n_ctxlens++] = v;
            tok = strtok(NULL, ",");
        }
        if (n_ctxlens == 0) { fprintf(stderr, "no context lengths given\n"); return 2; }
    }

    static const char *EXPECT_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707";
    static Gpt2Weights w;
    struct timespec sha0, sha1, load1;
    clock_gettime(CLOCK_MONOTONIC, &sha0);
    char actual_sha[65];
    if (sha256_file(safetensors_path, actual_sha)) { fprintf(stderr, "checkpoint not found: %s\n", safetensors_path); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &sha1);
    if (strcmp(actual_sha, EXPECT_SHA)) { fprintf(stderr, "fingerprint mismatch: computed=%s expected=%s\n", actual_sha, EXPECT_SHA); return 1; }
    if (gpt2_load_weights(&w, safetensors_path, NULL) /* already fingerprint-checked above; skip redundant re-hash */) { fprintf(stderr, "weight load failed\n"); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &load1);
    double sha_ms = ns_between(sha0, sha1) / 1e6;
    double load_ms = ns_between(sha1, load1) / 1e6;

    struct timespec setup0, setup1;
    clock_gettime(CLOCK_MONOTONIC, &setup0);
    for (int L = 0; L < GPT2_NLAYER; L++) if (setup_layer_attn_graph(&g_lag[L])) { fprintf(stderr, "graph setup failed\n"); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &setup1);
    double graph_setup_ms = ns_between(setup0, setup1) / 1e6;

    FILE *tsv = fopen(out_tsv_path, "w");
    if (!tsv) { fprintf(stderr, "cannot open %s for writing\n", out_tsv_path); return 1; }
    fprintf(tsv, "ctxlen\trep\tis_warmup\torder_first\tmode\tstep_ns\ttoken\tlogits_max_abs_diff\tprocess_peak_rss_kb\n");

    /* fixed, deterministic dummy context -- valid token ids, no tokenizer needed */
    static int ctx_tokens[GPT2_MAXT];
    for (int i = 0; i < GPT2_MAXT; i++) ctx_tokens[i] = (i * 37 + 101) % 50257;

    long kv_cache_bytes_per_layer_capacity = (long)MAXCACHE * 768 * 2 * (long)sizeof(float);
    printf("PROVENANCE sha_verify_ms=%.3f weight_load_ms=%.3f graph_setup_ms=%.3f kv_cache_bytes_per_layer_capacity=%ld kv_cache_bytes_total_capacity=%ld\n",
           sha_ms, load_ms, graph_setup_ms, kv_cache_bytes_per_layer_capacity, kv_cache_bytes_per_layer_capacity * GPT2_NLAYER);

    for (int ci = 0; ci < n_ctxlens; ci++) {
        int L = ctxlens[ci];
        for (int rep = 0; rep < warmup + reps; rep++) {
            int is_warmup = rep < warmup;
            int cached_first = (rep % 2) == 0;

            /* Compute each mode EXACTLY ONCE per repetition (timed), then
             * cross-check the two already-computed results -- no redundant
             * recomputation. `order` only controls which mode is computed
             * FIRST within the rep (for thermal/drift symmetry), not how
             * many times each is computed. */
            float cached_logits[GPT2_VOCAB], full_logits[GPT2_VOCAB];
            int cached_token = -1, full_token = -1;
            double cached_ns = 0, full_ns = 0;

            for (int order = 0; order < 2; order++) {
                int do_cached = (order == 0) ? cached_first : !cached_first;
                struct timespec s0, s1;
                if (do_cached) {
                    for (int LL = 0; LL < GPT2_NLAYER; LL++) reset_layer_attn_graph(&g_lag[LL]);
                    gpt2_prefill_cached_upto(&w, ctx_tokens, L, NULL);
                    clock_gettime(CLOCK_MONOTONIC, &s0);
                    if (gpt2_decode_step_cached(&w, ctx_tokens[L - 1], L - 1, cached_logits)) { fprintf(stderr, "cached decode failed at ctxlen=%d\n", L); return 1; }
                    clock_gettime(CLOCK_MONOTONIC, &s1);
                    cached_token = greedy_argmax(cached_logits, GPT2_VOCAB);
                    cached_ns = ns_between(s0, s1);
                } else {
                    clock_gettime(CLOCK_MONOTONIC, &s0);
                    gpt2_forward_ex(&w, ctx_tokens, L, full_logits, 1);
                    clock_gettime(CLOCK_MONOTONIC, &s1);
                    full_token = greedy_argmax(full_logits, GPT2_VOCAB);
                    full_ns = ns_between(s0, s1);
                }
            }

            if (cached_token != full_token) {
                fprintf(stderr, "TOKEN MISMATCH at ctxlen=%d rep=%d: cached=%d full=%d -- aborting, timing data would be over an unverified computation\n",
                        L, rep, cached_token, full_token);
                return 1;
            }
            float max_abs = 0;
            for (int v = 0; v < GPT2_VOCAB; v++) { float d = fabsf(cached_logits[v] - full_logits[v]); if (d > max_abs) max_abs = d; }
            if (max_abs > 1e-3f) {
                fprintf(stderr, "LOGITS MISMATCH at ctxlen=%d rep=%d: max_abs=%.6g -- aborting\n", L, rep, max_abs);
                return 1;
            }

            /* process_peak_rss_kb is the WHOLE-PROCESS high-water mark
             * (getrusage RUSAGE_SELF), monotonically non-decreasing across
             * the entire run -- it is NOT attributable to either mode
             * individually (both share the same process and the same
             * loaded weight buffers). It is recorded for completeness only;
             * do not read the cached-row and full-row values as a
             * per-mode memory comparison. The one genuinely mode-specific
             * memory number is the KV-cache capacity reported in the
             * PROVENANCE line (kv_cache_bytes_total_capacity) -- full
             * recompute allocates no equivalent persistent buffer. */
            long rss = peak_rss_kb();
            fprintf(tsv, "%d\t%d\t%d\t%d\tcached\t%.1f\t%d\t%.9g\t%ld\n", L, rep, is_warmup, cached_first, cached_ns, cached_token, max_abs, rss);
            fprintf(tsv, "%d\t%d\t%d\t%d\tfull\t%.1f\t%d\t%.9g\t%ld\n", L, rep, is_warmup, cached_first, full_ns, full_token, max_abs, rss);
        }
        printf("ctxlen=%d: %d warmup + %d measured reps x 2 modes done\n", L, warmup, reps);
    }
    fclose(tsv);
    puts("bench_algorithm: all samples token-verified and logits-verified against the other mode at every single measurement; raw data written to TSV");
    return 0;
}
