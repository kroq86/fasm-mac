/* Reproduces bench_sustained.c's EXACT live incremental cached path
 * (prefill via gpt2_prefill_cached for the prompt, then sequential
 * gpt2_decode_step_cached calls one at a time -- same functions, copied
 * verbatim) and at EVERY step compares its logits against a FRESH
 * gpt2_forward_ex full recompute over the identical token history, to
 * find the first position where they genuinely diverge (not just at the
 * end, as bench_sustained.c currently does). Prints max_abs_diff and
 * both greedy tokens at every step. */
#include "tensor_semantic_compiler.h"
#include "tensor_gpt2_forward.h"
#include <stdio.h>
#include <stdlib.h>

#if M != 768 || H != 12 || D != 64 || QW != 2304 || MAXCACHE < 128
#error "requires -DM=768 -DH=12 -DD=64 -DQW=2304 -DMAXCACHE>=128"
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

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s SAFETENSORS_PATH GEN_LEN\n", argv[0]); return 2; }
    const char *safetensors_path = argv[1];
    int gen_len = atoi(argv[2]);
    static const char *EXPECT_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707";
    static Gpt2Weights w;
    if (gpt2_load_weights(&w, safetensors_path, EXPECT_SHA)) { fprintf(stderr, "load failed\n"); return 1; }

    int prompt[4] = {464, 3290, 3332, 2159};
    int prompt_len = 4;
    for (int L = 0; L < GPT2_NLAYER; L++) if (setup_layer_attn_graph(&g_lag[L])) { fprintf(stderr, "graph setup failed\n"); return 1; }

    int seq[GPT2_MAXT];
    for (int i = 0; i < prompt_len; i++) seq[i] = prompt[i];

    float logits[GPT2_VOCAB];
    gpt2_prefill_cached(&w, seq, prompt_len - 1);
    if (gpt2_decode_step_cached(&w, seq[prompt_len - 1], prompt_len - 1, logits)) { fprintf(stderr, "first decode failed\n"); return 1; }
    seq[prompt_len] = greedy_argmax(logits, GPT2_VOCAB);

    /* full recompute reference at the same position, using the identical token so far */
    float logits_full[GPT2_VOCAB];
    gpt2_forward_ex(&w, seq, prompt_len, logits_full, 1);
    int tok_full0 = greedy_argmax(logits_full, GPT2_VOCAB);
    float m0 = 0; for (int v = 0; v < GPT2_VOCAB; v++) { float d = fabsf(logits[v]-logits_full[v]); if (d>m0) m0=d; }
    printf("step=0 pos=%d live_cached_token=%d full_token=%d max_abs=%.9g%s\n", prompt_len-1, seq[prompt_len], tok_full0, m0, seq[prompt_len]!=tok_full0 ? "  <-- MISMATCH" : "");

    for (int step = 1; step < gen_len; step++) {
        int pos = prompt_len + step - 1;
        if (gpt2_decode_step_cached(&w, seq[pos], pos, logits)) { fprintf(stderr, "decode failed at step=%d\n", step); return 1; }
        int tok_live = greedy_argmax(logits, GPT2_VOCAB);

        /* full recompute over the identical prefix the LIVE cached run has actually produced so far */
        gpt2_forward_ex(&w, seq, pos + 1, logits_full, 1);
        int tok_full = greedy_argmax(logits_full, GPT2_VOCAB);
        float m = 0; for (int v = 0; v < GPT2_VOCAB; v++) { float d = fabsf(logits[v]-logits_full[v]); if (d>m) m=d; }

        printf("step=%d pos=%d live_cached_token=%d full_token=%d max_abs=%.9g%s\n", step, pos, tok_live, tok_full, m, tok_live!=tok_full ? "  <-- MISMATCH" : "");

        seq[pos + 1] = tok_live; /* live cached run's own trajectory continues on its own choice, exactly like bench_sustained.c */
        if (tok_live != tok_full) {
            fprintf(stderr, "STOPPING at first divergence, step=%d pos=%d\n", step, pos);
            break;
        }
    }
    return 0;
}
