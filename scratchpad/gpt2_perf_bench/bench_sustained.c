/* Sustained end-to-end generation: TTFT + per-token decode latency +
 * tokens/sec over one continuous run (prompt -> N generated tokens),
 * for cached and full-recompute separately. Complements
 * bench_algorithm.c's isolated per-context-length marginal-step
 * measurements with what a real generation actually experiences: one
 * prefill, then a CONTINUOUS decode loop (cached: cache persists across
 * steps as in real use; full: context genuinely grows each step, same
 * as the generation gates).
 *
 * Self-consistency correctness bar (see bench_algorithm.c's header for
 * why that's adequate here): the two modes' full generated token
 * sequences are compared at the end; a mismatch aborts before any
 * timing is trusted enough to print.
 *
 * Fails closed: gen_len is bounds-checked before running.
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
static void gpt2_prefill_cached(const Gpt2Weights *w, const int *token_ids, int n) {
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
    static float lnf[GPT2_MAXT * GPT2_M];
    gpt2_layernorm_raw(h, n, GPT2_M, lnf);
    static float discard[GPT2_VOCAB];
    gpt2_matmul_bias(lnf + (n - 1) * GPT2_M, 1, GPT2_M, w->lnf_w_folded, w->lnf_b_folded, GPT2_VOCAB, discard);
}
static int gpt2_decode_step_cached(const Gpt2Weights *w, int token_id, int pos, float *logits_out) {
    float h[GPT2_M];
    for (int m = 0; m < GPT2_M; m++) h[m] = w->wte[token_id * GPT2_M + m] + w->wpe[pos * GPT2_M + m];
    for (int L = 0; L < GPT2_NLAYER; L++) {
        const Gpt2BlockWeights *bw = &w->blk[L];
        float ln1[GPT2_M]; gpt2_layernorm_raw(h, 1, GPT2_M, ln1);
        float qkv[GPT2_QW]; gpt2_matmul_bias(ln1, 1, GPT2_M, bw->attn_w, bw->attn_b, GPT2_QW, qkv);
        memcpy(g_lag[L].x, qkv, sizeof(float) * QW);
        if (tensor_transformer_steps_execute(g_lag[L].steps, g_lag[L].count)) return -1;
        float attn[GPT2_M]; memcpy(attn, g_lag[L].out, sizeof(float) * 768);
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
static int greedy_argmax(const float *logits, int vocab) { int b = 0; for (int i = 1; i < vocab; i++) if (logits[i] > logits[b]) b = i; return b; }
static double ms_between(struct timespec a, struct timespec b) { return (double)(b.tv_sec - a.tv_sec) * 1000.0 + (double)(b.tv_nsec - a.tv_nsec) / 1e6; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s SAFETENSORS_PATH OUT_TSV GEN_LEN\n", argv[0]); return 2; }
    const char *safetensors_path = argv[1];
    const char *out_tsv = argv[2];
    int gen_len = atoi(argv[3]);
    if (gen_len < 2 || gen_len > GPT2_MAXT - 4) { fprintf(stderr, "gen_len out of bounds [2,%d]: %d\n", GPT2_MAXT - 4, gen_len); return 2; }

    static const char *EXPECT_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707";
    static Gpt2Weights w;
    if (gpt2_load_weights(&w, safetensors_path, EXPECT_SHA)) { fprintf(stderr, "load failed\n"); return 1; }

    int prompt[4] = {464, 3290, 3332, 2159};
    int prompt_len = 4;

    /* --- cached run --- */
    for (int L = 0; L < GPT2_NLAYER; L++) if (setup_layer_attn_graph(&g_lag[L])) { fprintf(stderr, "graph setup failed\n"); return 1; }
    int seq_cached[GPT2_MAXT];
    for (int i = 0; i < prompt_len; i++) seq_cached[i] = prompt[i];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    gpt2_prefill_cached(&w, seq_cached, prompt_len - 1);
    float logits[GPT2_VOCAB];
    if (gpt2_decode_step_cached(&w, seq_cached[prompt_len - 1], prompt_len - 1, logits)) { fprintf(stderr, "first decode failed\n"); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ttft_cached_ms = ms_between(t0, t1);
    seq_cached[prompt_len] = greedy_argmax(logits, GPT2_VOCAB);

    double *cached_step_ms = malloc(sizeof(double) * (size_t)gen_len);
    for (int step = 1; step < gen_len; step++) {
        int pos = prompt_len + step - 1;
        struct timespec s0, s1;
        clock_gettime(CLOCK_MONOTONIC, &s0);
        if (gpt2_decode_step_cached(&w, seq_cached[pos], pos, logits)) { fprintf(stderr, "decode failed\n"); return 1; }
        clock_gettime(CLOCK_MONOTONIC, &s1);
        cached_step_ms[step] = ms_between(s0, s1);
        seq_cached[pos + 1] = greedy_argmax(logits, GPT2_VOCAB);
    }

    /* --- full-recompute run --- */
    int seq_full[GPT2_MAXT];
    for (int i = 0; i < prompt_len; i++) seq_full[i] = prompt[i];
    clock_gettime(CLOCK_MONOTONIC, &t0);
    gpt2_forward_ex(&w, seq_full, prompt_len, logits, 1);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ttft_full_ms = ms_between(t0, t1);
    seq_full[prompt_len] = greedy_argmax(logits, GPT2_VOCAB);

    double *full_step_ms = malloc(sizeof(double) * (size_t)gen_len);
    for (int step = 1; step < gen_len; step++) {
        int n = prompt_len + step;
        struct timespec s0, s1;
        clock_gettime(CLOCK_MONOTONIC, &s0);
        gpt2_forward_ex(&w, seq_full, n, logits, 1);
        clock_gettime(CLOCK_MONOTONIC, &s1);
        full_step_ms[step] = ms_between(s0, s1);
        seq_full[n] = greedy_argmax(logits, GPT2_VOCAB);
    }

    /* correctness: full generated sequences must match exactly before any timing is trusted */
    for (int i = 0; i < prompt_len + gen_len; i++) {
        if (seq_cached[i] != seq_full[i]) {
            fprintf(stderr, "SEQUENCE MISMATCH at position %d: cached=%d full=%d\n", i, seq_cached[i], seq_full[i]);
            /* DEBUG: recompute both modes fresh, from a common prefix ending at i-1, to inspect the divergent logits directly */
            int common[GPT2_MAXT];
            for (int k = 0; k < i; k++) common[k] = seq_cached[k];
            float dbg_full[GPT2_VOCAB];
            gpt2_forward_ex(&w, common, i, dbg_full, 1);
            for (int L = 0; L < GPT2_NLAYER; L++) if (setup_layer_attn_graph(&g_lag[L])) return 1;
            gpt2_prefill_cached(&w, common, i - 1); /* populates cache for positions 0..i-2 */
            float dbg_cached[GPT2_VOCAB];
            gpt2_decode_step_cached(&w, common[i - 1], i - 1, dbg_cached); /* decode position i-1 -> logits predicting position i */
            int tf = greedy_argmax(dbg_full, GPT2_VOCAB), tc = greedy_argmax(dbg_cached, GPT2_VOCAB);
            fprintf(stderr, "DEBUG recompute-from-common-prefix: full_token=%d cached_token=%d full_logit[full]=%.9g full_logit[cached]=%.9g cached_logit[full]=%.9g cached_logit[cached]=%.9g\n",
                    tf, tc, dbg_full[tf], dbg_full[tc], dbg_cached[tf], dbg_cached[tc]);
            float max_abs = 0; for (int v = 0; v < GPT2_VOCAB; v++) { float d = fabsf(dbg_full[v]-dbg_cached[v]); if (d>max_abs) max_abs=d; }
            fprintf(stderr, "DEBUG max_abs_diff_at_divergence_point=%.9g\n", max_abs);
            return 1;
        }
    }

    double cached_sum = 0, full_sum = 0;
    for (int step = 1; step < gen_len; step++) { cached_sum += cached_step_ms[step]; full_sum += full_step_ms[step]; }
    double cached_decode_s = cached_sum / 1000.0, full_decode_s = full_sum / 1000.0;

    FILE *tsv = fopen(out_tsv, "w");
    if (!tsv) { fprintf(stderr, "cannot open %s\n", out_tsv); return 1; }
    fprintf(tsv, "mode\tstep\tcontext_len\tstep_ms\n");
    fprintf(tsv, "cached\t0\t%d\t%.3f\n", prompt_len, ttft_cached_ms);
    fprintf(tsv, "full\t0\t%d\t%.3f\n", prompt_len, ttft_full_ms);
    for (int step = 1; step < gen_len; step++) {
        fprintf(tsv, "cached\t%d\t%d\t%.3f\n", step, prompt_len + step, cached_step_ms[step]);
        fprintf(tsv, "full\t%d\t%d\t%.3f\n", step, prompt_len + step, full_step_ms[step]);
    }
    fclose(tsv);

    printf("SUSTAINED prompt_len=%d gen_len=%d (both sequences byte-identical, self-consistency verified)\n", prompt_len, gen_len);
    printf("SUSTAINED cached: ttft_ms=%.3f decode_tokens=%d decode_s=%.6f tokens_per_sec=%.3f\n", ttft_cached_ms, gen_len - 1, cached_decode_s, (double)(gen_len - 1) / cached_decode_s);
    printf("SUSTAINED full:   ttft_ms=%.3f decode_tokens=%d decode_s=%.6f tokens_per_sec=%.3f\n", ttft_full_ms, gen_len - 1, full_decode_s, (double)(gen_len - 1) / full_decode_s);
    free(cached_step_ms); free(full_step_ms);
    return 0;
}
