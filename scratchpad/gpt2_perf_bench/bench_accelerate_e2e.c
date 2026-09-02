/* Accelerate SGEMM wired into the REAL cached-generation path (same
 * gpt2_prefill_cached/gpt2_decode_step_cached shape as bench_sustained.c
 * and tensorctl_gpt2.c -- prefill prompt_len-1, decode the rest one step
 * at a time, canonical CAUSAL_ATTENTION_CACHED executor unchanged), not
 * an isolated matmul-shape microbenchmark. Runs the SAME real GPT-2 124M
 * generation twice -- once with every gpt2_matmul_bias call replaced by
 * an Accelerate-backed equivalent, once scalar (baseline, unmodified) --
 * and compares full generated token sequences + logits before any timing
 * is trusted. Attention itself (the canonical executor) is identical in
 * both runs; only the surrounding projections/FFN/LM-head matmuls swap
 * backend. No claim about Metal/MPS/planner/general performance.
 *
 * This does not modify fasm/examples/tensorctl_gpt2.c or
 * tensor_gpt2_forward.h -- it is a standalone comparison harness reusing
 * the identical real weights and the identical prefill/decode pattern
 * those files use, built in scratchpad to avoid touching shared/
 * committed production files without a separate, explicit integration
 * decision.
 */
/* Accelerate.h's own headers use bare identifiers M/N/K as parameter
 * names (e.g. BLAS's sgemm_(... M, N, K ...)) -- these collide with the
 * canonical compiler's bare M/H/D/QW/F macros if -DM=768 etc. is passed
 * on the command line (defined before any #include). Fix: include
 * Accelerate.h FIRST, then #define the canonical compiler's macros in
 * source, then include the canonical compiler header. */
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define M 768
#define H 12
#define D 64
#define QW 2304
#define F 3072
#define MAXCACHE 128
#include "tensor_semantic_compiler.h"
#include "tensor_gpt2_forward.h"

#if GPT2_MAXT < 128
#error "requires -DGPT2_MAXT>=128"
#endif

static void gpt2_matmul_bias_sgemm(const float *a, int n, int k, const float *w, const float *b, int outdim, float *out) {
    for (int i = 0; i < n; i++) memcpy(&out[i * outdim], b, sizeof(float) * (size_t)outdim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, outdim, k, 1.0f, a, k, w, outdim, 1.0f, out, outdim);
}

typedef struct {
    Node g[3]; ExecStep steps[8]; Context ctx[8]; uint32_t count;
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

/* backend==0: scalar gpt2_matmul_bias (baseline, unmodified). backend==1:
 * Accelerate gpt2_matmul_bias_sgemm. Attention (canonical executor) is
 * identical either way. */
static void gpt2_prefill_cached_be(const Gpt2Weights *w, const int *token_ids, int n, int backend) {
    static float h[GPT2_MAXT * GPT2_M];
    for (int t = 0; t < n; t++) for (int m = 0; m < GPT2_M; m++)
        h[t * GPT2_M + m] = w->wte[token_ids[t] * GPT2_M + m] + w->wpe[t * GPT2_M + m];
    static float ln1[GPT2_MAXT * GPT2_M], qkv[GPT2_MAXT * GPT2_QW], attn[GPT2_MAXT * GPT2_M];
    static float proj[GPT2_MAXT * GPT2_M], res1[GPT2_MAXT * GPT2_M], ln2[GPT2_MAXT * GPT2_M];
    static float fc[GPT2_MAXT * GPT2_F], act[GPT2_MAXT * GPT2_F], fcproj[GPT2_MAXT * GPT2_M];
    void (*mm)(const float *, int, int, const float *, const float *, int, float *) = backend ? gpt2_matmul_bias_sgemm : gpt2_matmul_bias;
    for (int L = 0; L < GPT2_NLAYER; L++) {
        const Gpt2BlockWeights *bw = &w->blk[L];
        gpt2_layernorm_raw(h, n, GPT2_M, ln1);
        mm(ln1, n, GPT2_M, bw->attn_w, bw->attn_b, GPT2_QW, qkv);
        gpt2_causal_attention(qkv, n, attn);
        for (int t = 0; t < n; t++) {
            memcpy(&g_lag[L].cache_data[t * 768], &qkv[t * GPT2_QW + GPT2_M], sizeof(float) * 768);
            memcpy(&g_lag[L].cache_aux[t * 768], &qkv[t * GPT2_QW + 2 * GPT2_M], sizeof(float) * 768);
        }
        g_lag[L].g[LAG_CACHE].tensor.aux_count = (uint32_t)n;
        mm(attn, n, GPT2_M, bw->projw, bw->projb, GPT2_M, proj);
        for (int i = 0; i < n * GPT2_M; i++) res1[i] = h[i] + proj[i];
        gpt2_layernorm_raw(res1, n, GPT2_M, ln2);
        mm(ln2, n, GPT2_M, bw->fc_w, bw->fc_b, GPT2_F, fc);
        gpt2_gelu(fc, n * GPT2_F, act);
        mm(act, n, GPT2_F, bw->fcproj_w, bw->fcproj_b, GPT2_M, fcproj);
        for (int i = 0; i < n * GPT2_M; i++) h[i] = res1[i] + fcproj[i];
    }
}
static int gpt2_decode_step_cached_be(const Gpt2Weights *w, int token_id, int pos, float *logits_out, int backend) {
    void (*mm)(const float *, int, int, const float *, const float *, int, float *) = backend ? gpt2_matmul_bias_sgemm : gpt2_matmul_bias;
    float h[GPT2_M];
    for (int m = 0; m < GPT2_M; m++) h[m] = w->wte[token_id * GPT2_M + m] + w->wpe[pos * GPT2_M + m];
    for (int L = 0; L < GPT2_NLAYER; L++) {
        const Gpt2BlockWeights *bw = &w->blk[L];
        float ln1[GPT2_M]; gpt2_layernorm_raw(h, 1, GPT2_M, ln1);
        float qkv[GPT2_QW]; mm(ln1, 1, GPT2_M, bw->attn_w, bw->attn_b, GPT2_QW, qkv);
        memcpy(g_lag[L].x, qkv, sizeof(float) * QW);
        if (tensor_transformer_steps_execute(g_lag[L].steps, g_lag[L].count)) return -1;
        float attn[GPT2_M]; memcpy(attn, g_lag[L].out, sizeof(float) * 768);
        float proj[GPT2_M]; mm(attn, 1, GPT2_M, bw->projw, bw->projb, GPT2_M, proj);
        float res1[GPT2_M]; for (int i = 0; i < GPT2_M; i++) res1[i] = h[i] + proj[i];
        float ln2[GPT2_M]; gpt2_layernorm_raw(res1, 1, GPT2_M, ln2);
        float fc[GPT2_F]; mm(ln2, 1, GPT2_M, bw->fc_w, bw->fc_b, GPT2_F, fc);
        float act[GPT2_F]; gpt2_gelu(fc, GPT2_F, act);
        float fcproj[GPT2_M]; mm(act, 1, GPT2_F, bw->fcproj_w, bw->fcproj_b, GPT2_M, fcproj);
        for (int i = 0; i < GPT2_M; i++) h[i] = res1[i] + fcproj[i];
    }
    float lnf[GPT2_M]; gpt2_layernorm_raw(h, 1, GPT2_M, lnf);
    mm(lnf, 1, GPT2_M, w->lnf_w_folded, w->lnf_b_folded, GPT2_VOCAB, logits_out);
    return 0;
}
static int greedy_argmax(const float *logits, int vocab) { int b = 0; for (int i = 1; i < vocab; i++) if (logits[i] > logits[b]) b = i; return b; }
static double ms_between(struct timespec a, struct timespec b) { return (double)(b.tv_sec - a.tv_sec) * 1000.0 + (double)(b.tv_nsec - a.tv_nsec) / 1e6; }

/* runs one full generation with the given backend; returns tokens/sec and ttft via out params */
static void run_generation(const Gpt2Weights *w, const int *prompt, int prompt_len, int gen_len, int backend,
                            int *seq_out, double *ttft_ms_out, double *tokens_per_sec_out) {
    for (int L = 0; L < GPT2_NLAYER; L++) if (setup_layer_attn_graph(&g_lag[L])) { fprintf(stderr, "graph setup failed\n"); exit(1); }
    for (int i = 0; i < prompt_len; i++) seq_out[i] = prompt[i];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    gpt2_prefill_cached_be(w, seq_out, prompt_len - 1, backend);
    float logits[GPT2_VOCAB];
    if (gpt2_decode_step_cached_be(w, seq_out[prompt_len - 1], prompt_len - 1, logits, backend)) { fprintf(stderr, "first decode failed\n"); exit(1); }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *ttft_ms_out = ms_between(t0, t1);
    seq_out[prompt_len] = greedy_argmax(logits, GPT2_VOCAB);

    double sum_ms = 0;
    for (int step = 1; step < gen_len; step++) {
        int pos = prompt_len + step - 1;
        struct timespec s0, s1;
        clock_gettime(CLOCK_MONOTONIC, &s0);
        if (gpt2_decode_step_cached_be(w, seq_out[pos], pos, logits, backend)) { fprintf(stderr, "decode failed\n"); exit(1); }
        clock_gettime(CLOCK_MONOTONIC, &s1);
        sum_ms += ms_between(s0, s1);
        seq_out[pos + 1] = greedy_argmax(logits, GPT2_VOCAB);
    }
    *tokens_per_sec_out = (double)(gen_len - 1) / (sum_ms / 1000.0);
}

int main(int argc, char **argv) {
    const char *safetensors_path = argc > 1 ? argv[1] : "/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors";
    int gen_len = argc > 2 ? atoi(argv[2]) : 40;
    if (gen_len < 2 || gen_len > GPT2_MAXT - 4) { fprintf(stderr, "gen_len out of bounds [2,%d]: %d\n", GPT2_MAXT - 4, gen_len); return 2; }
    static const char *EXPECT_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707";
    static Gpt2Weights w;
    if (gpt2_load_weights(&w, safetensors_path, EXPECT_SHA)) { fprintf(stderr, "load failed\n"); return 1; }

    int prompt[4] = {464, 3290, 3332, 2159};
    int prompt_len = 4;

    static int seq_scalar[GPT2_MAXT], seq_sgemm[GPT2_MAXT];
    double ttft_scalar, tps_scalar, ttft_sgemm, tps_sgemm;
    run_generation(&w, prompt, prompt_len, gen_len, 0, seq_scalar, &ttft_scalar, &tps_scalar);
    run_generation(&w, prompt, prompt_len, gen_len, 1, seq_sgemm, &ttft_sgemm, &tps_sgemm);

    for (int i = 0; i < prompt_len + gen_len; i++) {
        if (seq_scalar[i] != seq_sgemm[i]) {
            fprintf(stderr, "SEQUENCE MISMATCH at position %d: scalar=%d sgemm=%d -- ABORTING, no timing trusted\n", i, seq_scalar[i], seq_sgemm[i]);
            return 1;
        }
    }
    printf("E2E prompt_len=%d gen_len=%d backend=scalar vs backend=accelerate_sgemm (generated sequences byte-identical, %d tokens)\n", prompt_len, gen_len, prompt_len + gen_len);
    printf("E2E scalar:    ttft_ms=%.3f tokens_per_sec=%.3f\n", ttft_scalar, tps_scalar);
    printf("E2E sgemm:     ttft_ms=%.3f tokens_per_sec=%.3f\n", ttft_sgemm, tps_sgemm);
    printf("E2E speedup:   ttft=%.2fx sustained_tokens_per_sec=%.2fx\n", ttft_scalar / ttft_sgemm, tps_sgemm / tps_scalar);
    return 0;
}
