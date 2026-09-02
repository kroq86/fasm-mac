/* Accelerate SGEMM spike: does swapping this project's scalar
 * gpt2_matmul_bias for Apple's Accelerate cblas_sgemm change wall time on
 * the actual matmul shapes GPT-2 124M uses, using real layer-0 weights
 * and shape-compatible real-valued inputs, with every output verified
 * before any timing is trusted.
 *
 * Scope discipline (explicit): this measures ONE thing -- scalar C loop
 * vs. Accelerate SGEMM, same shapes, same machine, same process. It makes
 * NO claim about Metal, MPS, a planner, or the overall model's
 * performance. It is a narrow backend-swap-cost experiment, not a
 * production integration.
 *
 * Input honesty: `ln1` (LayerNorm(embedding) for a fixed synthetic token
 * sequence, real layer-0 weights) is used as the input for qkv_proj,
 * out_proj, AND ffn_expand. It is the mathematically correct input only
 * for qkv_proj. For out_proj (whose real input is the attention output)
 * and ffn_expand (whose real input is ln2, LayerNorm of the
 * post-attention residual), `ln1` is a SHAPE-COMPATIBLE PROXY -- same
 * dimensions and the same real distributional family (a real LayerNorm
 * output), but not the actual tensor that operation consumes in a real
 * forward pass. ffn_proj's input (`fc_act`) IS the real, correctly-
 * chained GELU(fc(ln1)) activation. This matters for correctness-diff
 * interpretation (the specific float values differ from a real pass) but
 * not for the timing comparison itself, which depends on shape/dtype/
 * memory layout, not data values -- BLAS GEMM timing is data-independent
 * by construction. lm_head's input is also the `ln1` proxy (real input
 * would be the final block's post-ln_f hidden state).
 *
 * Method: warmup=3 discarded, 10 measured repetitions per (category,
 * ctxlen), scalar/sgemm order alternated every repetition. Every single
 * rep's max-abs output diff is recorded and checked for NaN/Inf (a
 * plain `diff > TOL` comparison would silently PASS a NaN, since any
 * comparison against NaN is false in IEEE754) and against a 1e-3 bound
 * (matching this project's other cached-generation gates) before being
 * kept; a violation aborts the run. All 10 raw per-rep timings are kept,
 * not just the median -- see raw_accelerate_sgemm.tsv.
 */
#include "tensor_gpt2_forward.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define WARMUP 3
#define REPS 10
#define TOL 1e-3f

static double ns_between(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e9 + (double)(b.tv_nsec - a.tv_nsec);
}

/* out[n,outdim] = a[n,k] @ w[k,outdim] + b[outdim], row-major -- identical
 * math to gpt2_matmul_bias, via Accelerate. */
static void sgemm_bias(const float *a, int n, int k, const float *w, const float *b, int outdim, float *out) {
    for (int i = 0; i < n; i++) memcpy(&out[i * outdim], b, sizeof(float) * (size_t)outdim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, outdim, k, 1.0f, a, k, w, outdim, 1.0f, out, outdim);
}

typedef struct { double scalar_ns[REPS], sgemm_ns[REPS]; float max_abs_diff; } Result;

static void run_category(FILE *raw_tsv, const char *name, const float *a, int n, int k, const float *w, const float *b, int outdim, Result *r) {
    float *out_scalar = malloc(sizeof(float) * (size_t)n * outdim);
    float *out_sgemm = malloc(sizeof(float) * (size_t)n * outdim);
    r->max_abs_diff = 0;
    /* warmup, discarded */
    for (int i = 0; i < WARMUP; i++) { gpt2_matmul_bias(a, n, k, w, b, outdim, out_scalar); sgemm_bias(a, n, k, w, b, outdim, out_sgemm); }
    for (int rep = 0; rep < REPS; rep++) {
        struct timespec t0, t1;
        int scalar_first = (rep % 2 == 0);
        if (scalar_first) {
            clock_gettime(CLOCK_MONOTONIC, &t0); gpt2_matmul_bias(a, n, k, w, b, outdim, out_scalar); clock_gettime(CLOCK_MONOTONIC, &t1);
            r->scalar_ns[rep] = ns_between(t0, t1);
            clock_gettime(CLOCK_MONOTONIC, &t0); sgemm_bias(a, n, k, w, b, outdim, out_sgemm); clock_gettime(CLOCK_MONOTONIC, &t1);
            r->sgemm_ns[rep] = ns_between(t0, t1);
        } else {
            clock_gettime(CLOCK_MONOTONIC, &t0); sgemm_bias(a, n, k, w, b, outdim, out_sgemm); clock_gettime(CLOCK_MONOTONIC, &t1);
            r->sgemm_ns[rep] = ns_between(t0, t1);
            clock_gettime(CLOCK_MONOTONIC, &t0); gpt2_matmul_bias(a, n, k, w, b, outdim, out_scalar); clock_gettime(CLOCK_MONOTONIC, &t1);
            r->scalar_ns[rep] = ns_between(t0, t1);
        }
        float rep_max = 0;
        int saw_nan_or_inf = 0;
        for (int v = 0; v < n * outdim; v++) {
            if (!isfinite(out_scalar[v]) || !isfinite(out_sgemm[v])) { saw_nan_or_inf = 1; break; }
            float d = fabsf(out_scalar[v] - out_sgemm[v]);
            if (d > rep_max) rep_max = d;
        }
        if (saw_nan_or_inf) {
            fprintf(stderr, "%s n=%d rep=%d: NaN/Inf in output -- ABORTING\n", name, n, rep);
            exit(1);
        }
        if (rep_max > r->max_abs_diff) r->max_abs_diff = rep_max;
        if (rep_max > TOL) {
            fprintf(stderr, "%s n=%d: OUTPUT MISMATCH rep=%d max_abs_diff=%.6g exceeds bound %.6g -- ABORTING\n", name, n, rep, rep_max, TOL);
            exit(1);
        }
        fprintf(raw_tsv, "%s\t%d\t%d\t%d\t%.3f\t%.3f\t%.9g\n", name, n, rep, scalar_first, r->scalar_ns[rep] / 1e6, r->sgemm_ns[rep] / 1e6, rep_max);
    }
    free(out_scalar); free(out_sgemm);
}

static double median(double *v, int n) {
    double tmp[REPS]; memcpy(tmp, v, sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (tmp[j] < tmp[i]) { double t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t; }
    return n % 2 ? tmp[n / 2] : (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0;
}

int main(int argc, char **argv) {
    const char *safetensors_path = argc > 1 ? argv[1] : "/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors";
    const char *raw_tsv_path = argc > 2 ? argv[2] : "raw_accelerate_sgemm.tsv";
    static const char *EXPECT_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707";
    static Gpt2Weights w;
    if (gpt2_load_weights(&w, safetensors_path, EXPECT_SHA)) { fprintf(stderr, "load failed\n"); return 1; }
    FILE *raw_tsv = fopen(raw_tsv_path, "w");
    if (!raw_tsv) { fprintf(stderr, "cannot open %s\n", raw_tsv_path); return 1; }
    fprintf(raw_tsv, "category\tn\trep\tscalar_first\tscalar_ms\tsgemm_ms\tmax_abs_diff\n");

    static int tokens[128];
    for (int i = 0; i < 128; i++) tokens[i] = (i * 37 + 101) % 50257;
    static float h[128 * GPT2_M], ln1[128 * GPT2_M];
    for (int t = 0; t < 128; t++) for (int m = 0; m < GPT2_M; m++) h[t * GPT2_M + m] = w.wte[tokens[t] * GPT2_M + m] + w.wpe[t * GPT2_M + m];
    gpt2_layernorm_raw(h, 128, GPT2_M, ln1);

    static float fc_act[128 * GPT2_F];
    { static float fc[128 * GPT2_F]; gpt2_matmul_bias(ln1, 128, GPT2_M, w.blk[0].fc_w, w.blk[0].fc_b, GPT2_F, fc); gpt2_gelu(fc, 128 * GPT2_F, fc_act); }

    printf("%-14s %6s %10s %10s %8s %6s %14s\n", "category", "n", "scalar_ms", "sgemm_ms", "speedup", "reps", "max_abs_diff");
    int ctxlens[] = {4, 16, 32, 64};
    struct { const char *name; int k, outdim; const float *a, *w_, *b; int fixed_n; } cats[] = {
        {"qkv_proj", GPT2_M, GPT2_QW, ln1, w.blk[0].attn_w, w.blk[0].attn_b, 0},
        {"out_proj", GPT2_M, GPT2_M, ln1, w.blk[0].projw, w.blk[0].projb, 0},
        {"ffn_expand", GPT2_M, GPT2_F, ln1, w.blk[0].fc_w, w.blk[0].fc_b, 0},
        {"ffn_proj", GPT2_F, GPT2_M, fc_act, w.blk[0].fcproj_w, w.blk[0].fcproj_b, 0},
        {"lm_head", GPT2_M, GPT2_VOCAB, ln1, w.lnf_w_folded, w.lnf_b_folded, 1},
    };
    for (size_t c = 0; c < sizeof(cats) / sizeof(cats[0]); c++) {
        if (cats[c].fixed_n) {
            Result r; run_category(raw_tsv, cats[c].name, cats[c].a, 1, cats[c].k, cats[c].w_, cats[c].b, cats[c].outdim, &r);
            double sm = median(r.scalar_ns, REPS) / 1e6, gm = median(r.sgemm_ns, REPS) / 1e6;
            printf("%-14s %6d %10.4f %10.4f %8.2fx %6d %14.6g\n", cats[c].name, 1, sm, gm, sm / gm, REPS, r.max_abs_diff);
            continue;
        }
        for (size_t i = 0; i < sizeof(ctxlens) / sizeof(ctxlens[0]); i++) {
            int n = ctxlens[i];
            Result r; run_category(raw_tsv, cats[c].name, cats[c].a, n, cats[c].k, cats[c].w_, cats[c].b, cats[c].outdim, &r);
            double sm = median(r.scalar_ns, REPS) / 1e6, gm = median(r.sgemm_ns, REPS) / 1e6;
            printf("%-14s %6d %10.4f %10.4f %8.2fx %6d %14.6g\n", cats[c].name, n, sm, gm, sm / gm, REPS, r.max_abs_diff);
        }
    }
    fclose(raw_tsv);
    return 0;
}
