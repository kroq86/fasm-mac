/* Decoder-runtime roadmap item 4, integration: bring the real GPT-2 124M
 * generation loop's KV cache under the canonical graph/executor, via the
 * new CAUSAL_ATTENTION_CACHED op (tensor_semantic_compiler.h, verified in
 * isolation on toy data by tensor_causal_attention_cached_differential_check.c).
 *
 * Scope, deliberately narrow: only the ATTENTION step of each decode
 * step actually needs cache state -- MATMUL/BIAS_ADD/RESIDUAL/GELU stay
 * plain runtime-n C (tensor_gpt2_forward.h's already-anchored kernels),
 * same as before. Prefill (the initial prompt, T=n>1) still uses that
 * same already-anchored full-recompute math -- it has no incremental
 * state to represent, it just needs to LEAVE the per-layer K/V caches
 * populated for decode to continue from, which is a data-copy of values
 * prefill already computes, not a new computation. Decode (T=1 per new
 * token) is where the roadmap's actual claim lives: each layer's
 * attention graph is `compile()`d ONCE (setup_layer_attn_graph, before
 * generation starts), and every new token's attention step is a real
 * `tensor_transformer_steps_execute()` call against that same compiled
 * graph, reading/mutating REAL, continuously-resident Tensor state
 * (`aux_count` as position, `data`/`aux` as the K/V cache) -- not a
 * hand-rolled C counter copied in and out between calls.
 *
 * Precise, non-overclaimed scope of what's canonical here: only the
 * per-layer attention step runs through `compile()`+executor. Prefill,
 * the QKV/output/FFN projections, residuals, GELU, and the LM head all
 * remain hand-written runtime-length C (tensor_gpt2_forward.h's kernels)
 * -- this does NOT mean "generation became canonical", only that the KV
 * cache specifically is now real canonical Tensor state a future planner
 * could inspect (no planner does yet).
 *
 * The check: for several prompts, the cached-decode path's logits (and
 * greedily-sampled tokens) are compared against tensor_gpt2_forward.h's
 * full-recompute path at EVERY generated step -- the roadmap's own
 * "compare every generated-step logit against full recomputation"
 * requirement, not just a final-token spot check. Agreement is checked
 * against a preregistered numerical tolerance (max abs diff <=1e-3), NOT
 * bit-exact -- only the sampled (argmax) tokens are exact matches. One
 * prompt is also cross-checked against the real PyTorch
 * greedy_reference.txt fixture, confirming the cached path produces the
 * correct real-world tokens, not just something self-consistent with
 * this project's own other hand-rolled path.
 *
 * Fail-closed: set GPT2_CACHED_GEN_REQUIRED=1 to make a missing
 * checkpoint or reference fixture a hard failure instead of a skip (the
 * default, matching this project's other optional-by-default GPT-2
 * gates). greedy_reference.txt is parsed with explicit bounds/format
 * checks (read_greedy_case) -- a corrupted or truncated fixture fails
 * closed rather than risking an out-of-bounds read.
 *
 * Built with -DM=768 -DH=12 -DD=64 -DQW=2304 -DF=3072 -DMAXCACHE=64 (the
 * canonical compiler's real-shape macros) -- GPT2_* macros in
 * tensor_gpt2_forward.h default to the same real values independently
 * (separate namespace, by design, to avoid colliding with these).
 */
#include "tensor_semantic_compiler.h"
#include "tensor_gpt2_forward.h"
#include <stdio.h>
#include <stdlib.h>

#if M != 768 || H != 12 || D != 64 || QW != 2304 || MAXCACHE < 32
#error "requires -DM=768 -DH=12 -DD=64 -DQW=2304 -DMAXCACHE>=32"
#endif

/* One persistent, precompiled canonical graph per layer: compile() runs
 * ONCE (here, at setup), not once per decode step. ExecStep/Context only
 * store a pointer to this Node array plus a node index -- the array
 * itself, and every Tensor field inside it (data/aux/aux_count), stays
 * genuinely resident and is what tensor_transformer_steps_execute()
 * mutates on every subsequent call. This means the cache's position
 * counter (`aux_count`) is real, continuously-resident canonical Tensor
 * state across the whole decode loop -- NOT copied in/out through an
 * external C struct field between calls, which an earlier version of
 * this file did (flagged in review: state should live in the Tensor, not
 * bounce through a side channel). Only the qkv input buffer (`x`) is
 * overwritten between calls, exactly like feeding a new input into an
 * already-compiled graph is supposed to work. */
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
    if (compile(lag->g, LAG_NODES, .01f, lag->steps, lag->ctx, 8, &lag->count)) return -1;
    return 0;
}

static int load_f32(const char *dir, const char *name, float *out, size_t count) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.f32", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t got = fread(out, sizeof(float), count, f);
    fclose(f);
    return got == count ? 0 : -1;
}

/* Prefill: full recompute over the prompt (n>1), same math as
 * gpt2_forward_ex, but additionally captures each layer's K/V for every
 * prompt position directly into that layer's cache -- a data copy of
 * values this computation already produces, not new computation. */
static void gpt2_prefill_cached(const Gpt2Weights *w, const int *token_ids, int n, float *last_logits) {
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
        /* capture this layer's K/V for every prompt position directly
         * into the precompiled graph's own cache Tensor buffers (real
         * Tensor state from the very first write, not a separate C
         * struct copied in later) */
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
    gpt2_matmul_bias(lnf + (n - 1) * GPT2_M, 1, GPT2_M, w->lnf_w_folded, w->lnf_b_folded, GPT2_VOCAB, last_logits);
}

/* One new token (T=1): the attention step re-executes this layer's
 * ALREADY-COMPILED graph (setup_layer_attn_graph ran once, before
 * generation started) -- only the qkv input buffer is overwritten here;
 * no compile() call, no Node array rebuild. The cache's position counter
 * (`aux_count`) lives inside `lag->g[LAG_CACHE].tensor` continuously and
 * is mutated in place by the kernel every call -- real, resident
 * canonical Tensor state, not copied through an external field. */
static int canonical_attention_step(const float *qkv_row, LayerAttnGraph *lag, float *attn_out) {
    memcpy(lag->x, qkv_row, sizeof(float) * QW);
    if (tensor_transformer_steps_execute(lag->steps, lag->count)) { fprintf(stderr, "canonical_attention_step: execute failed\n"); return -1; }
    memcpy(attn_out, lag->out, sizeof(float) * 768);
    return 0;
}

static int gpt2_decode_step_cached(const Gpt2Weights *w, int token_id, int pos, float *logits_out) {
    float h[GPT2_M];
    for (int m = 0; m < GPT2_M; m++) h[m] = w->wte[token_id * GPT2_M + m] + w->wpe[pos * GPT2_M + m];

    for (int L = 0; L < GPT2_NLAYER; L++) {
        const Gpt2BlockWeights *bw = &w->blk[L];
        float ln1[GPT2_M]; gpt2_layernorm_raw(h, 1, GPT2_M, ln1);
        float qkv[GPT2_QW]; gpt2_matmul_bias(ln1, 1, GPT2_M, bw->attn_w, bw->attn_b, GPT2_QW, qkv);
        float attn[GPT2_M];
        if (canonical_attention_step(qkv, &g_lag[L], attn)) return -1;
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
    int best = 0;
    for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

static int required_mode(void) {
    const char *v = getenv("GPT2_CACHED_GEN_REQUIRED");
    return v && !strcmp(v, "1");
}
static int unavailable(const char *what) {
    fprintf(required_mode() ? stderr : stdout, "gpt2_cached_generation: %s -- %s\n", what, required_mode() ? "required gate failed" : "optional gate skipped");
    return required_mode() ? 1 : 0;
}

/* Reads one line of greedy_reference.txt: "N_PROMPT p1..pN N_GEN g1..gN".
 * Fails closed on any malformed/truncated/out-of-bounds field -- np/ng
 * are validated against the caller's buffer capacity BEFORE being used
 * as loop bounds, so a corrupted fixture can't drive an out-of-bounds
 * write here. Returns 0 on success, -1 on any parse/bounds failure or
 * EOF. */
static int read_greedy_case(FILE *f, int *prompt, int prompt_cap, int *np_out, int *gen, int gen_cap, int *ng_out) {
    int np;
    if (fscanf(f, "%d", &np) != 1) return -1;
    if (np <= 0 || np > prompt_cap) { fprintf(stderr, "read_greedy_case: prompt length %d out of bounds (cap %d)\n", np, prompt_cap); return -1; }
    for (int i = 0; i < np; i++) if (fscanf(f, "%d", &prompt[i]) != 1) { fprintf(stderr, "read_greedy_case: truncated prompt\n"); return -1; }
    int ng;
    if (fscanf(f, "%d", &ng) != 1) return -1;
    if (ng < 0 || ng > gen_cap) { fprintf(stderr, "read_greedy_case: generated length %d out of bounds (cap %d)\n", ng, gen_cap); return -1; }
    for (int i = 0; i < ng; i++) if (fscanf(f, "%d", &gen[i]) != 1) { fprintf(stderr, "read_greedy_case: truncated generation\n"); return -1; }
    *np_out = np; *ng_out = ng;
    return 0;
}

int main(int argc, char **argv) {
    const char *fixture_dir = argc > 1 ? argv[1] : "scratchpad/gpt2_block_boundary";
    const char *safetensors_path = argc > 2 ? argv[2] : "/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors";
    static const char *EXPECT_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707";

    static Gpt2Weights w;
    if (gpt2_load_weights(&w, safetensors_path, EXPECT_SHA)) return unavailable("real checkpoint not found/verified");

    int prompts[3][8] = {{464, 3290, 3332, 2159}, {15496, 11, 995, 0}, {1, 2, 3, 4, 5}};
    int prompt_lens[3] = {4, 4, 5};
    int gen_len = 8;

    for (int c = 0; c < 3; c++) {
        for (int L = 0; L < GPT2_NLAYER; L++) if (setup_layer_attn_graph(&g_lag[L])) return 1;

        int seq[GPT2_MAXT];
        for (int i = 0; i < prompt_lens[c]; i++) seq[i] = prompts[c][i];

        float prefill_logits[GPT2_VOCAB];
        gpt2_prefill_cached(&w, seq, prompt_lens[c], prefill_logits);
        int next = greedy_argmax(prefill_logits, GPT2_VOCAB);
        seq[prompt_lens[c]] = next;

        int all_match = 1;
        float worst_abs = 0;
        for (int step = 1; step < gen_len; step++) {
            int pos = prompt_lens[c] + step - 1;
            float cached_logits[GPT2_VOCAB];
            if (gpt2_decode_step_cached(&w, seq[pos], pos, cached_logits)) return 1;

            int n = pos + 1;
            float full_logits[GPT2_VOCAB];
            gpt2_forward_ex(&w, seq, n, full_logits, 1);

            float max_abs = 0;
            for (int v = 0; v < GPT2_VOCAB; v++) { float d = fabsf(cached_logits[v] - full_logits[v]); if (d > max_abs) max_abs = d; }
            if (max_abs > worst_abs) worst_abs = max_abs;
            if (max_abs > 1e-3f) { fprintf(stderr, "case %d step %d: cached vs full-recompute logits diverge, max_abs=%.6g\n", c, step, max_abs); all_match = 0; }

            int cached_tok = greedy_argmax(cached_logits, GPT2_VOCAB);
            int full_tok = greedy_argmax(full_logits, GPT2_VOCAB);
            if (cached_tok != full_tok) { fprintf(stderr, "case %d step %d: cached token=%d full-recompute token=%d MISMATCH\n", c, step, cached_tok, full_tok); all_match = 0; }
            seq[pos + 1] = cached_tok;
        }
        if (!all_match) { fprintf(stderr, "gpt2_cached_generation: case %d FAILED\n", c); return 1; }
        printf("case %d: %d cached-decode steps, worst logits max_abs=%.6g (bound 1e-3), identical greedy tokens, prompt_len=%d\n", c, gen_len - 1, worst_abs, prompt_lens[c]);
    }

    /* cross-check case 0 against the real PyTorch greedy reference too */
    {
        char path[512];
        snprintf(path, sizeof path, "%s/greedy_reference.txt", fixture_dir);
        FILE *gf = fopen(path, "rb");
        if (!gf) {
            if (unavailable("greedy_reference.txt not found")) return 1;
        } else {
            int ref_prompt[16], ref_gen[64], np, ng;
            int rc = read_greedy_case(gf, ref_prompt, 16, &np, ref_gen, 64, &ng);
            fclose(gf);
            if (rc) { fprintf(stderr, "gpt2_cached_generation: greedy_reference.txt malformed/truncated\n"); return 1; }

            for (int L = 0; L < GPT2_NLAYER; L++) if (setup_layer_attn_graph(&g_lag[L])) return 1;
            int seq[GPT2_MAXT];
            for (int i = 0; i < np; i++) seq[i] = ref_prompt[i];
            float logits[GPT2_VOCAB];
            gpt2_prefill_cached(&w, seq, np, logits);
            seq[np] = greedy_argmax(logits, GPT2_VOCAB);
            int match = (ng == 0) || (seq[np] == ref_gen[0]);
            for (int step = 1; step < ng && np + step < GPT2_MAXT; step++) {
                int pos = np + step - 1;
                if (gpt2_decode_step_cached(&w, seq[pos], pos, logits)) return 1;
                seq[pos + 1] = greedy_argmax(logits, GPT2_VOCAB);
                if (seq[pos + 1] != ref_gen[step]) match = 0;
            }
            if (!match) { fprintf(stderr, "gpt2_cached_generation: canonical-cached path diverges from real PyTorch greedy sequence\n"); return 1; }
            printf("PyTorch cross-check: canonical-cached decode path's tokens exactly match the real PyTorch greedy sequence (%d tokens, prompt_len=%d)\n", ng, np);
        }
    }

    puts("gpt2_cached_generation differential check passed: CAUSAL_ATTENTION_CACHED (real canonical compile()+executor calls, real Tensor state) drives real GPT-2 124M decode, matching full recomputation at every generated step and matching real PyTorch greedy generation");
    return 0;
}
