/* Per-op wall-clock profiler for GPT-2 124M forward passes on this
 * project's own scalar-C + canonical-executor path (fasm-mac,
 * x86_64/Rosetta). NOT a cross-runtime or cross-implementation claim.
 *
 * Claim framing (calibration-first rule):
 *  - Unknown being tested: what fraction of wall time, in THIS project's
 *    own hybrid implementation, goes to LayerNorm / QKV projection /
 *    attention / output projection / FFN expansion (matmul+GELU) / FFN
 *    projection / LM head, at prefill ctxlen 4/16/32/64 and at one
 *    cached decode step.
 *  - Baseline/prior art: "transformers are matmul-bound, LM head is
 *    disproportionately expensive due to large vocab" is the standard,
 *    expected pattern (llm.c's own timing, PyTorch-profiler traces of
 *    GPT-2-scale models, Karpathy's build-nanogpt). No novel measurement
 *    technique here -- this is engineering measurement, not a claim of
 *    discovery.
 *  - Falsifier: if projections+LM head are NOT the dominant share
 *    (<~50%), that contradicts the expected matmul-bound pattern and
 *    points at something else (LayerNorm, buffer copies, or this
 *    project's own executor/ExecStep dispatch overhead) dominating.
 *  - Decision link: dominant projections/LM head -> next spike is
 *    Accelerate SGEMM. Dominant executor-dispatch overhead -> next step
 *    is profiling ExecStep dispatch itself, not swapping BLAS.
 *  - What's genuinely unknown (not implied by the classical claim): the
 *    classical claim describes FLOP distribution, not wall-clock time in
 *    THIS specific hand-rolled-C + canonical-executor hybrid, which has
 *    extra per-call overhead (Node array setup, executor dispatch,
 *    buffer copies) the classical intuition doesn't predict the size of.
 *
 * Correctness: reuses the project's own already-anchored gpt2_forward_ex
 * and the already bit-exact-verified cached decode path; this file only
 * adds timers around existing, already-correct computations. It does not
 * re-derive or re-verify GPT-2 correctness.
 */
#include "tensor_semantic_compiler.h"
#include "tensor_gpt2_forward.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if M != 768 || H != 12 || D != 64 || QW != 2304 || MAXCACHE < 128
#error "requires -DM=768 -DH=12 -DD=64 -DQW=2304 -DMAXCACHE>=128"
#endif
#if GPT2_MAXT < 128
#error "requires -DGPT2_MAXT>=128"
#endif

enum { CAT_LN, CAT_QKV, CAT_ATTN, CAT_OUTPROJ, CAT_FFN_EXPAND, CAT_FFN_PROJ, CAT_LMHEAD, CAT_COUNT };
static const char *CAT_NAME[CAT_COUNT] = {"layernorm", "qkv_proj", "attention", "out_proj", "ffn_expand", "ffn_proj", "lm_head"};

static double ns_between(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e9 + (double)(b.tv_nsec - a.tv_nsec);
}
#define TIC(t) clock_gettime(CLOCK_MONOTONIC, &(t))
#define ACC(cat, t0, t1) (acc[cat] += ns_between((t0), (t1)))

/* Forces the optimizer to keep every timed computation's result live --
 * without this, buffers written but never read (esp. the LM head's
 * 50257-wide logits, computed once and otherwise unused here) get
 * dead-store/dead-code eliminated under -O2, silently reporting ~0ms for
 * real work. Found by inspection: lm_head read exactly 0.000ms at every
 * context length in the first run, which is not physically plausible for
 * a 768x50257 matmul comparable in cost to the FFN matmuls. */
static volatile float g_sink = 0;
static void sink_array(const float *a, int n) { float s = 0; for (int i = 0; i < n; i += (n > 64 ? n / 64 : 1)) s += a[i]; g_sink += s; }

/* Mirrors gpt2_forward_ex's internal per-layer computation exactly (same
 * calls, same order) but with timers bracketing each named phase. n
 * tokens, batched, full (non-cached) attention -- this IS the real
 * prefill/full-recompute path already used and validated elsewhere. */
static void profile_prefill(const Gpt2Weights *w, const int *token_ids, int n, double acc[CAT_COUNT]) {
    static float h[GPT2_MAXT * GPT2_M];
    for (int t = 0; t < n; t++) for (int m = 0; m < GPT2_M; m++)
        h[t * GPT2_M + m] = w->wte[token_ids[t] * GPT2_M + m] + w->wpe[t * GPT2_M + m];
    static float ln1[GPT2_MAXT * GPT2_M], qkv[GPT2_MAXT * GPT2_QW], attn[GPT2_MAXT * GPT2_M];
    static float proj[GPT2_MAXT * GPT2_M], res1[GPT2_MAXT * GPT2_M], ln2[GPT2_MAXT * GPT2_M];
    static float fc[GPT2_MAXT * GPT2_F], act[GPT2_MAXT * GPT2_F], fcproj[GPT2_MAXT * GPT2_M];
    struct timespec t0, t1;
    for (int L = 0; L < GPT2_NLAYER; L++) {
        const Gpt2BlockWeights *bw = &w->blk[L];
        TIC(t0); gpt2_layernorm_raw(h, n, GPT2_M, ln1); TIC(t1); ACC(CAT_LN, t0, t1); sink_array(ln1, n * GPT2_M);
        TIC(t0); gpt2_matmul_bias(ln1, n, GPT2_M, bw->attn_w, bw->attn_b, GPT2_QW, qkv); TIC(t1); ACC(CAT_QKV, t0, t1); sink_array(qkv, n * GPT2_QW);
        TIC(t0); gpt2_causal_attention(qkv, n, attn); TIC(t1); ACC(CAT_ATTN, t0, t1); sink_array(attn, n * GPT2_M);
        TIC(t0); gpt2_matmul_bias(attn, n, GPT2_M, bw->projw, bw->projb, GPT2_M, proj); TIC(t1); ACC(CAT_OUTPROJ, t0, t1); sink_array(proj, n * GPT2_M);
        for (int i = 0; i < n * GPT2_M; i++) res1[i] = h[i] + proj[i];
        TIC(t0); gpt2_layernorm_raw(res1, n, GPT2_M, ln2); TIC(t1); ACC(CAT_LN, t0, t1); sink_array(ln2, n * GPT2_M);
        TIC(t0);
        gpt2_matmul_bias(ln2, n, GPT2_M, bw->fc_w, bw->fc_b, GPT2_F, fc);
        gpt2_gelu(fc, n * GPT2_F, act);
        TIC(t1); ACC(CAT_FFN_EXPAND, t0, t1); sink_array(act, n * GPT2_F);
        TIC(t0); gpt2_matmul_bias(act, n, GPT2_F, bw->fcproj_w, bw->fcproj_b, GPT2_M, fcproj); TIC(t1); ACC(CAT_FFN_PROJ, t0, t1); sink_array(fcproj, n * GPT2_M);
        for (int i = 0; i < n * GPT2_M; i++) h[i] = res1[i] + fcproj[i];
    }
    static float lnf[GPT2_MAXT * GPT2_M];
    static float logits[GPT2_VOCAB];
    TIC(t0); gpt2_layernorm_raw(h, n, GPT2_M, lnf); TIC(t1); ACC(CAT_LN, t0, t1); sink_array(lnf, n * GPT2_M);
    TIC(t0); gpt2_matmul_bias(lnf + (n - 1) * GPT2_M, 1, GPT2_M, w->lnf_w_folded, w->lnf_b_folded, GPT2_VOCAB, logits); TIC(t1); ACC(CAT_LMHEAD, t0, t1); sink_array(logits, GPT2_VOCAB);
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
static void prefill_cache_only(const Gpt2Weights *w, const int *token_ids, int n) {
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
}
/* Mirrors gpt2_decode_step_cached exactly, with timers. */
static int profile_decode_step(const Gpt2Weights *w, int token_id, int pos, double acc[CAT_COUNT]) {
    struct timespec t0, t1;
    float h[GPT2_M];
    for (int m = 0; m < GPT2_M; m++) h[m] = w->wte[token_id * GPT2_M + m] + w->wpe[pos * GPT2_M + m];
    for (int L = 0; L < GPT2_NLAYER; L++) {
        const Gpt2BlockWeights *bw = &w->blk[L];
        float ln1[GPT2_M]; TIC(t0); gpt2_layernorm_raw(h, 1, GPT2_M, ln1); TIC(t1); ACC(CAT_LN, t0, t1); sink_array(ln1, GPT2_M);
        float qkv[GPT2_QW]; TIC(t0); gpt2_matmul_bias(ln1, 1, GPT2_M, bw->attn_w, bw->attn_b, GPT2_QW, qkv); TIC(t1); ACC(CAT_QKV, t0, t1); sink_array(qkv, GPT2_QW);
        memcpy(g_lag[L].x, qkv, sizeof(float) * QW);
        TIC(t0); if (tensor_transformer_steps_execute(g_lag[L].steps, g_lag[L].count)) return -1; TIC(t1); ACC(CAT_ATTN, t0, t1);
        float attn[GPT2_M]; memcpy(attn, g_lag[L].out, sizeof(float) * 768); sink_array(attn, GPT2_M);
        float proj[GPT2_M]; TIC(t0); gpt2_matmul_bias(attn, 1, GPT2_M, bw->projw, bw->projb, GPT2_M, proj); TIC(t1); ACC(CAT_OUTPROJ, t0, t1); sink_array(proj, GPT2_M);
        float res1[GPT2_M]; for (int i = 0; i < GPT2_M; i++) res1[i] = h[i] + proj[i];
        float ln2[GPT2_M]; TIC(t0); gpt2_layernorm_raw(res1, 1, GPT2_M, ln2); TIC(t1); ACC(CAT_LN, t0, t1); sink_array(ln2, GPT2_M);
        float fc[GPT2_F]; float act[GPT2_F];
        TIC(t0);
        gpt2_matmul_bias(ln2, 1, GPT2_M, bw->fc_w, bw->fc_b, GPT2_F, fc);
        gpt2_gelu(fc, GPT2_F, act);
        TIC(t1); ACC(CAT_FFN_EXPAND, t0, t1); sink_array(act, GPT2_F);
        float fcproj[GPT2_M]; TIC(t0); gpt2_matmul_bias(act, 1, GPT2_F, bw->fcproj_w, bw->fcproj_b, GPT2_M, fcproj); TIC(t1); ACC(CAT_FFN_PROJ, t0, t1); sink_array(fcproj, GPT2_M);
        for (int i = 0; i < GPT2_M; i++) h[i] = res1[i] + fcproj[i];
    }
    float lnf[GPT2_M]; static float logits[GPT2_VOCAB];
    TIC(t0); gpt2_layernorm_raw(h, 1, GPT2_M, lnf); TIC(t1); ACC(CAT_LN, t0, t1); sink_array(lnf, GPT2_M);
    TIC(t0); gpt2_matmul_bias(lnf, 1, GPT2_M, w->lnf_w_folded, w->lnf_b_folded, GPT2_VOCAB, logits); TIC(t1); ACC(CAT_LMHEAD, t0, t1); sink_array(logits, GPT2_VOCAB);
    return 0;
}

static void print_table(const char *label, double acc[CAT_COUNT]) {
    double total = 0; for (int c = 0; c < CAT_COUNT; c++) total += acc[c];
    printf("\n%s (total %.3fms)\n", label, total / 1e6);
    printf("  %-12s %10s %8s\n", "category", "ms", "pct");
    for (int c = 0; c < CAT_COUNT; c++)
        printf("  %-12s %10.3f %7.1f%%\n", CAT_NAME[c], acc[c] / 1e6, total > 0 ? 100.0 * acc[c] / total : 0.0);
}

int main(int argc, char **argv) {
    const char *safetensors_path = argc > 1 ? argv[1] : "/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors";
    static const char *EXPECT_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707";
    static Gpt2Weights w;
    if (gpt2_load_weights(&w, safetensors_path, EXPECT_SHA)) { fprintf(stderr, "load failed\n"); return 1; }

    static int tokens[128];
    for (int i = 0; i < 128; i++) tokens[i] = (i * 37 + 101) % 50257;

    int ctxlens[] = {4, 16, 32, 64};
    for (size_t i = 0; i < sizeof(ctxlens) / sizeof(ctxlens[0]); i++) {
        int n = ctxlens[i];
        double acc[CAT_COUNT] = {0};
        profile_prefill(&w, tokens, n, acc); /* warmup / cache warm */
        for (int c = 0; c < CAT_COUNT; c++) acc[c] = 0;
        profile_prefill(&w, tokens, n, acc);
        char label[64]; snprintf(label, sizeof label, "PREFILL ctxlen=%d", n);
        print_table(label, acc);
    }

    for (int L = 0; L < GPT2_NLAYER; L++) if (setup_layer_attn_graph(&g_lag[L])) { fprintf(stderr, "graph setup failed\n"); return 1; }
    int prompt_len = 32;
    prefill_cache_only(&w, tokens, prompt_len - 1); /* cache holds positions 0..prompt_len-2, aux_count=prompt_len-1 */
    double acc[CAT_COUNT] = {0};
    /* untimed warmup step at pos=prompt_len-1 (appends one entry, aux_count=prompt_len) */
    if (profile_decode_step(&w, tokens[prompt_len - 1], prompt_len - 1, acc)) { fprintf(stderr, "decode failed\n"); return 1; }
    /* timed step at the NEXT position (pos=prompt_len) -- a distinct position, not a duplicate append */
    for (int c = 0; c < CAT_COUNT; c++) acc[c] = 0;
    if (profile_decode_step(&w, tokens[prompt_len], prompt_len, acc)) { fprintf(stderr, "decode failed\n"); return 1; }
    print_table("CACHED DECODE STEP (ctx=32 at time of step)", acc);

    return 0;
}
