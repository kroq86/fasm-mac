/* Real GPT-2 124M, all 12 blocks + ln_f + tied LM head, differential check.
 *
 * Extends tensor_gpt2_block0_differential_check.c's proven single-block
 * pipeline (see that file's header comment for the full provenance,
 * fingerprint, LayerNorm-fold rationale, and self-written-runtime
 * boundary -- identical here, not repeated). New in this file:
 *   - all 12 blocks chained (block i's output feeds block i+1's input,
 *     each with its own real weights, loaded the same fold-affine way);
 *   - the final `ln_f` LayerNorm, folded the same way into the tied LM
 *     head matmul;
 *   - the tied LM head: GPT-2 reuses `wte.weight` [VOCAB,M] as the output
 *     projection. This project's MATMUL needs [in,out] = [M,VOCAB], the
 *     OPPOSITE orientation, so `wte` is explicitly transposed once at
 *     load time into a second buffer -- a reported layout conversion, not
 *     a value edit, exactly as this project's self-written-runtime rule
 *     permits.
 *
 * Still no tokenizer, no generation, no training. Boundaries compared:
 * hidden state after block 5, after block 11 (the last), after ln_f, and
 * the final logits -- stopping diagnostics at the first one exceeding
 * tolerance. Tolerance is widened from the single-block gate to account
 * for 12x the sequential accumulation, preregistered before running:
 * abs(diff) < max(3e-2, 2e-3*|ref|).
 */
#include "tensor_semantic_compiler.h"
#include "tensor_safetensors_loader.h"
#include "tensor_sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GPT2_FIXTURE_DIR
#define GPT2_FIXTURE_DIR "scratchpad/gpt2_block_boundary"
#endif
#ifndef GPT2_SAFETENSORS_PATH
#define GPT2_SAFETENSORS_PATH "/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
#endif
#ifndef VOCAB
#define VOCAB 50257
#endif
#ifndef NLAYER
#define NLAYER 12
#endif
#ifndef MAXPOS
#define MAXPOS 1024
#endif

typedef struct {
    float ln1_w[M], ln1_b[M], ln2_w[M], ln2_b[M];
    float attn_w[M * QW], attn_b[QW];
    float projw[M * M], projb[M];
    float fc_w[M * F], fc_b[F];
    float fcproj_w[F * M], fcproj_b[M];
} BlockWeights;
static BlockWeights g_blk[NLAYER];

typedef struct {
    float ln1[T * M], gln1[T * M], ln1aux[2 * T];
    float gwqkv[M * QW], gbqkv[QW];
    float qkvmm[T * QW], gqkvmm[T * QW], qkv[T * QW], gqkv[T * QW];
    float att[H * T * D], gatt[H * T * D], attaux[H * T * T];
    float merged[T * M], gmerged[T * M];
    float gwproj[M * M], gbproj[M];
    float projmm[T * M], gprojmm[T * M], proj[T * M], gproj[T * M];
    float res1[T * M], gres1[T * M];
    float ln2[T * M], gln2[T * M], ln2aux[2 * T];
    float gwfc[M * F], gbfc[F];
    float fcmm[T * F], gfcmm[T * F], fc[T * F], gfc[T * F];
    float act[T * F], gact[T * F];
    float gwfcproj[F * M], gbfcproj[M];
    float fcprojmm[T * M], gfcprojmm[T * M], fcproj[T * M], gfcproj[T * M];
    float res2[T * M], gres2[T * M];
} BlockActs;
static BlockActs g_act[NLAYER];

static float g_wte[VOCAB * M], g_wte_t[M * VOCAB], g_wpe_full[MAXPOS * M];
static float g_lnf_w[M], g_lnf_b[M];
static float g_tok_ids[T], g_pos_ids[T];
static float g_ref_layer5[T * M], g_ref_layer11[T * M], g_ref_lnf[T * M], g_ref_logits[T * VOCAB];

static int load_f32(const char *dir, const char *name, float *out, size_t count) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.f32", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t got = fread(out, sizeof(float), count, f);
    fclose(f);
    return got == count ? 0 : -1;
}

static void fold_layernorm_affine(const float *gamma, const float *beta, float *w, float *b, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        double s = 0;
        for (int i = 0; i < in_dim; i++) s += (double)beta[i] * w[i * out_dim + o];
        b[o] += (float)s;
    }
    for (int i = 0; i < in_dim; i++) for (int o = 0; o < out_dim; o++) w[i * out_dim + o] *= gamma[i];
}

typedef struct { float max_abs, max_rel; int fails, nonfinite, worst_i; } BoundaryStat;

static BoundaryStat compare_boundary(const char *label, const float *got, const float *ref, int n) {
    BoundaryStat st = {0, 0, 0, 0, -1};
    float worst = -1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(got[i])) st.nonfinite++;
        float diff = fabsf(got[i] - ref[i]);
        float bound = fmaxf(3e-2f, 2e-3f * fabsf(ref[i]));
        if (diff > st.max_abs) st.max_abs = diff;
        float rel = diff / (fabsf(ref[i]) + 1e-8f);
        if (rel > st.max_rel) st.max_rel = rel;
        if (diff > bound) { st.fails++; if (diff > worst) { worst = diff; st.worst_i = i; } }
    }
    printf("boundary %-16s elements=%-8d max_abs=%.6g max_rel=%.6g nonfinite=%d fails=%d",
           label, n, st.max_abs, st.max_rel, st.nonfinite, st.fails);
    if (st.fails > 0) printf(" worst_at=%d got=%.9g ref=%.9g", st.worst_i, got[st.worst_i], ref[st.worst_i]);
    printf("\n");
    return st;
}

int main(int argc, char **argv) {
    if (M != 768 || H != 12 || D != 64 || QW != 2304 || F != 3072 || T != 8) {
        fprintf(stderr, "requires -DT=8 -DM=768 -DH=12 -DD=64 -DQW=2304 -DF=3072\n");
        return 1;
    }
    const char *fixture_dir = argc > 1 ? argv[1] : GPT2_FIXTURE_DIR;
    const char *safetensors_path = argc > 2 ? argv[2] : GPT2_SAFETENSORS_PATH;

    char actual_sha[65];
    if (sha256_file(safetensors_path, actual_sha)) {
        printf("gpt2_full: safetensors artifact not found at %s -- skipped\n", safetensors_path);
        return 0;
    }
    static const char *EXPECT_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707";
    if (strcmp(actual_sha, EXPECT_SHA)) {
        fprintf(stderr, "gpt2_full: artifact fingerprint mismatch: computed=%s expected=%s -- fail closed\n", actual_sha, EXPECT_SHA);
        return 1;
    }
    printf("gpt2_full: artifact fingerprint verified (sha256=%s)\n", actual_sha);

    SafetensorsFile st;
    if (st_open(&st, safetensors_path)) { fprintf(stderr, "gpt2_full: failed to open safetensors file\n"); return 1; }

    uint64_t s1[1] = {(uint64_t)M}, s2304[1] = {2304}, sMQW[2] = {(uint64_t)M, 2304}, sMM[2] = {(uint64_t)M, (uint64_t)M};
    uint64_t sF[1] = {(uint64_t)F}, sMF[2] = {(uint64_t)M, (uint64_t)F}, sFM[2] = {(uint64_t)F, (uint64_t)M};
    int ok = 0;
    char name[64];
    for (int L = 0; L < NLAYER; L++) {
        BlockWeights *bw = &g_blk[L];
        snprintf(name, sizeof name, "h.%d.ln_1.weight", L); ok |= st_read_f32(&st, name, s1, 1, bw->ln1_w);
        snprintf(name, sizeof name, "h.%d.ln_1.bias", L); ok |= st_read_f32(&st, name, s1, 1, bw->ln1_b);
        snprintf(name, sizeof name, "h.%d.ln_2.weight", L); ok |= st_read_f32(&st, name, s1, 1, bw->ln2_w);
        snprintf(name, sizeof name, "h.%d.ln_2.bias", L); ok |= st_read_f32(&st, name, s1, 1, bw->ln2_b);
        snprintf(name, sizeof name, "h.%d.attn.c_attn.weight", L); ok |= st_read_f32(&st, name, sMQW, 2, bw->attn_w);
        snprintf(name, sizeof name, "h.%d.attn.c_attn.bias", L); ok |= st_read_f32(&st, name, s2304, 1, bw->attn_b);
        snprintf(name, sizeof name, "h.%d.attn.c_proj.weight", L); ok |= st_read_f32(&st, name, sMM, 2, bw->projw);
        snprintf(name, sizeof name, "h.%d.attn.c_proj.bias", L); ok |= st_read_f32(&st, name, s1, 1, bw->projb);
        snprintf(name, sizeof name, "h.%d.mlp.c_fc.weight", L); ok |= st_read_f32(&st, name, sMF, 2, bw->fc_w);
        snprintf(name, sizeof name, "h.%d.mlp.c_fc.bias", L); ok |= st_read_f32(&st, name, sF, 1, bw->fc_b);
        snprintf(name, sizeof name, "h.%d.mlp.c_proj.weight", L); ok |= st_read_f32(&st, name, sFM, 2, bw->fcproj_w);
        snprintf(name, sizeof name, "h.%d.mlp.c_proj.bias", L); ok |= st_read_f32(&st, name, s1, 1, bw->fcproj_b);
        if (ok) { fprintf(stderr, "gpt2_full: block %d failed to load -- fail closed\n", L); return 1; }
    }
    uint64_t sVM[2] = {(uint64_t)VOCAB, (uint64_t)M}, sPM[2] = {(uint64_t)MAXPOS, (uint64_t)M};
    ok |= st_read_f32(&st, "wte.weight", sVM, 2, g_wte);
    ok |= st_read_f32(&st, "wpe.weight", sPM, 2, g_wpe_full);
    ok |= st_read_f32(&st, "ln_f.weight", s1, 1, g_lnf_w);
    ok |= st_read_f32(&st, "ln_f.bias", s1, 1, g_lnf_b);
    st_close(&st);
    if (ok) { fprintf(stderr, "gpt2_full: wte/wpe/ln_f failed to load -- fail closed\n"); return 1; }
    printf("gpt2_full: all %d blocks + wte/wpe/ln_f loaded directly from model.safetensors\n", NLAYER);

    /* explicit, reported layout conversion: wte is [VOCAB,M] (embedding-
     * table orientation); the tied LM head needs [M,VOCAB] (MATMUL's
     * [in,out] convention). transpose once into a second buffer -- the
     * original g_wte is left untouched since EMBED_LOOKUP still needs it
     * in its original orientation. */
    for (int v = 0; v < VOCAB; v++) for (int m = 0; m < M; m++) g_wte_t[m * VOCAB + v] = g_wte[v * M + m];
    printf("gpt2_full: wte transposed [%d,%d] -> [%d,%d] for the tied LM head matmul (layout conversion, no value edit)\n", VOCAB, M, M, VOCAB);

    if (load_f32(fixture_dir, "full_layer5_output", g_ref_layer5, T * M) ||
        load_f32(fixture_dir, "full_layer11_output", g_ref_layer11, T * M) ||
        load_f32(fixture_dir, "full_ln_f_output", g_ref_lnf, T * M) ||
        load_f32(fixture_dir, "full_logits", g_ref_logits, T * VOCAB)) {
        printf("gpt2_full: full-model reference tensors not found under %s -- skipped\n", fixture_dir);
        return 0;
    }

    for (int L = 0; L < NLAYER; L++) {
        fold_layernorm_affine(g_blk[L].ln1_w, g_blk[L].ln1_b, g_blk[L].attn_w, g_blk[L].attn_b, M, QW);
        fold_layernorm_affine(g_blk[L].ln2_w, g_blk[L].ln2_b, g_blk[L].fc_w, g_blk[L].fc_b, M, F);
    }
    static float lnf_w_folded[M * VOCAB], lnf_b_folded[VOCAB] = {0};
    memcpy(lnf_w_folded, g_wte_t, sizeof lnf_w_folded);
    fold_layernorm_affine(g_lnf_w, g_lnf_b, lnf_w_folded, lnf_b_folded, M, VOCAB);

    int fixed_tokens[8] = {464, 3290, 3332, 2159, 0, 1, 50256, 464};
    for (int i = 0; i < T; i++) { g_tok_ids[i] = (float)fixed_tokens[i]; g_pos_ids[i] = (float)i; }

    /* The canonical compiler caps a single graph at MAX_NODES=64 (a
     * shared, heavily-used constant this project leaves untouched here).
     * A full 12-block+ln_f+logits graph needs ~290 nodes, well over that,
     * so this runs as a SEQUENCE of small per-stage compile()+execute()
     * calls instead of one mega-graph -- each well under the cap, plain
     * float buffers threaded between calls. This needs no gradient/
     * backward, so splitting the forward pass this way changes nothing
     * about correctness, exactly like the KV-cache work's own per-step
     * compile() calls for a similarly fixed-size-graph constraint. */
    static float hidden[T * M];
    {
        enum { N_WTE, N_TOKID, N_WPE, N_POSID, N_TOKEMB, N_POSEMB, N_X0, EMB_NODES };
        static Node g[EMB_NODES]; static ExecStep steps[EMB_NODES * 4]; static Context ctx[EMB_NODES * 4];
        static float tokemb[T * M], gtokemb[T * M] = {0}, posemb[T * M], gposemb[T * M] = {0}, gx0[T * M] = {0};
        static float gwte[VOCAB * M] = {0}, gtokid[T] = {0}, gwpe[MAXPOS * M] = {0}, gposid[T] = {0};
        g[N_WTE] = (Node){LEAF, NONE, NONE, PARAM, {g_wte, gwte, NULL, VOCAB, M, 0}};
        g[N_TOKID] = (Node){LEAF, NONE, NONE, CONSTANT, {g_tok_ids, gtokid, NULL, 1, T, 0}};
        g[N_WPE] = (Node){LEAF, NONE, NONE, PARAM, {g_wpe_full, gwpe, NULL, MAXPOS, M, 0}};
        g[N_POSID] = (Node){LEAF, NONE, NONE, CONSTANT, {g_pos_ids, gposid, NULL, 1, T, 0}};
        g[N_TOKEMB] = (Node){EMBED_LOOKUP, N_WTE, N_TOKID, TEMP, {tokemb, gtokemb, NULL, T, M, 0}};
        g[N_POSEMB] = (Node){EMBED_LOOKUP, N_WPE, N_POSID, TEMP, {posemb, gposemb, NULL, T, M, 0}};
        g[N_X0] = (Node){RESIDUAL, N_TOKEMB, N_POSEMB, TEMP, {hidden, gx0, NULL, T, M, 0}};
        uint32_t count = 0;
        if (compile(g, EMB_NODES, .01f, steps, ctx, EMB_NODES * 4, &count)) { fprintf(stderr, "gpt2_full: embedding compile rejected\n"); return 1; }
        if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "gpt2_full: embedding execute failed\n"); return 1; }
    }

    static float layer5_out[T * M], layer11_out[T * M];
    for (int L = 0; L < NLAYER; L++) {
        BlockWeights *bw = &g_blk[L];
        BlockActs *ba = &g_act[0]; /* one shared activation scratch, reused every iteration */
        enum { X, LN1, WQKV, BQKV, QKVMM, QKV, ATT, MERGED, WPROJ, BPROJ, PROJMM, PROJ, RES1,
               LN2, WFC, BFC, FCMM, FC, ACT, WFCPROJ, BFCPROJ, FCPROJMM, FCPROJ, RES2, BLOCK_NODES };
        static Node g[BLOCK_NODES]; static ExecStep steps[BLOCK_NODES * 4]; static Context ctx[BLOCK_NODES * 4];
        static float gx[T * M] = {0};
        g[X] = (Node){LEAF, NONE, NONE, INPUT, {hidden, gx, NULL, T, M, 0}};
        g[LN1] = (Node){LAYERNORM, X, NONE, TEMP, {ba->ln1, ba->gln1, ba->ln1aux, T, M, 2 * T}};
        g[WQKV] = (Node){LEAF, NONE, NONE, PARAM, {bw->attn_w, ba->gwqkv, NULL, M, QW, 0}};
        g[BQKV] = (Node){LEAF, NONE, NONE, PARAM, {bw->attn_b, ba->gbqkv, NULL, 1, QW, 0}};
        g[QKVMM] = (Node){MATMUL, LN1, WQKV, TEMP, {ba->qkvmm, ba->gqkvmm, NULL, T, QW, 0}};
        g[QKV] = (Node){BIAS_ADD, QKVMM, BQKV, TEMP, {ba->qkv, ba->gqkv, NULL, T, QW, 0}};
        g[ATT] = (Node){CAUSAL_ATTENTION, QKV, NONE, TEMP, {ba->att, ba->gatt, ba->attaux, 1, H * T * D, H * T * T}};
        g[MERGED] = (Node){CONTIGUOUS, ATT, NONE, TEMP, {ba->merged, ba->gmerged, NULL, T, M, 0}};
        g[WPROJ] = (Node){LEAF, NONE, NONE, PARAM, {bw->projw, ba->gwproj, NULL, M, M, 0}};
        g[BPROJ] = (Node){LEAF, NONE, NONE, PARAM, {bw->projb, ba->gbproj, NULL, 1, M, 0}};
        g[PROJMM] = (Node){MATMUL, MERGED, WPROJ, TEMP, {ba->projmm, ba->gprojmm, NULL, T, M, 0}};
        g[PROJ] = (Node){BIAS_ADD, PROJMM, BPROJ, TEMP, {ba->proj, ba->gproj, NULL, T, M, 0}};
        g[RES1] = (Node){RESIDUAL, X, PROJ, TEMP, {ba->res1, ba->gres1, NULL, T, M, 0}};
        g[LN2] = (Node){LAYERNORM, RES1, NONE, TEMP, {ba->ln2, ba->gln2, ba->ln2aux, T, M, 2 * T}};
        g[WFC] = (Node){LEAF, NONE, NONE, PARAM, {bw->fc_w, ba->gwfc, NULL, M, F, 0}};
        g[BFC] = (Node){LEAF, NONE, NONE, PARAM, {bw->fc_b, ba->gbfc, NULL, 1, F, 0}};
        g[FCMM] = (Node){MATMUL, LN2, WFC, TEMP, {ba->fcmm, ba->gfcmm, NULL, T, F, 0}};
        g[FC] = (Node){BIAS_ADD, FCMM, BFC, TEMP, {ba->fc, ba->gfc, NULL, T, F, 0}};
        g[ACT] = (Node){GELU, FC, NONE, TEMP, {ba->act, ba->gact, NULL, T, F, 0}};
        g[WFCPROJ] = (Node){LEAF, NONE, NONE, PARAM, {bw->fcproj_w, ba->gwfcproj, NULL, F, M, 0}};
        g[BFCPROJ] = (Node){LEAF, NONE, NONE, PARAM, {bw->fcproj_b, ba->gbfcproj, NULL, 1, M, 0}};
        g[FCPROJMM] = (Node){MATMUL, ACT, WFCPROJ, TEMP, {ba->fcprojmm, ba->gfcprojmm, NULL, T, M, 0}};
        g[FCPROJ] = (Node){BIAS_ADD, FCPROJMM, BFCPROJ, TEMP, {ba->fcproj, ba->gfcproj, NULL, T, M, 0}};
        g[RES2] = (Node){RESIDUAL, RES1, FCPROJ, TEMP, {ba->res2, ba->gres2, NULL, T, M, 0}};
        uint32_t count = 0;
        if (compile(g, BLOCK_NODES, .01f, steps, ctx, BLOCK_NODES * 4, &count)) { fprintf(stderr, "gpt2_full: block %d compile rejected\n", L); return 1; }
        if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "gpt2_full: block %d execute failed\n", L); return 1; }
        memcpy(hidden, ba->res2, sizeof hidden);
        if (L == 5) memcpy(layer5_out, hidden, sizeof hidden);
        if (L == 11) memcpy(layer11_out, hidden, sizeof hidden);
    }

    static float g_lnf_out[T * M], g_logits[T * VOCAB];
    {
        enum { X, LNF, WTIED, BTIED, LOGITSMM, LOGITS, LNF_NODES };
        static Node g[LNF_NODES]; static ExecStep steps[LNF_NODES * 4]; static Context ctx[LNF_NODES * 4];
        static float gx[T * M] = {0}, glnf[T * M] = {0}, lnfaux[2 * T];
        static float glnfw[M * VOCAB] = {0}, glnfb[VOCAB] = {0}, glogitsmm[T * VOCAB] = {0}, glogits[T * VOCAB] = {0};
        g[X] = (Node){LEAF, NONE, NONE, INPUT, {hidden, gx, NULL, T, M, 0}};
        g[LNF] = (Node){LAYERNORM, X, NONE, TEMP, {g_lnf_out, glnf, lnfaux, T, M, 2 * T}};
        g[WTIED] = (Node){LEAF, NONE, NONE, PARAM, {lnf_w_folded, glnfw, NULL, M, VOCAB, 0}};
        g[BTIED] = (Node){LEAF, NONE, NONE, PARAM, {lnf_b_folded, glnfb, NULL, 1, VOCAB, 0}};
        g[LOGITSMM] = (Node){MATMUL, LNF, WTIED, TEMP, {g_logits, glogitsmm, NULL, T, VOCAB, 0}}; /* reused as scratch below */
        g[LOGITS] = (Node){BIAS_ADD, LOGITSMM, BTIED, TEMP, {g_logits, glogits, NULL, T, VOCAB, 0}};
        uint32_t count = 0;
        if (compile(g, LNF_NODES, .01f, steps, ctx, LNF_NODES * 4, &count)) { fprintf(stderr, "gpt2_full: ln_f/logits compile rejected\n"); return 1; }
        if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "gpt2_full: ln_f/logits execute failed\n"); return 1; }
    }

    /* g_lnf_out is the RAW (pre-affine) LAYERNORM output -- gamma/beta
     * were folded into the tied logits matmul instead of being applied
     * here (same reasoning as block0's ln_1/ln_2). Recompute the real
     * post-affine value purely to diff against the reference boundary,
     * same as block0 does; the actual logits computation already used
     * the fold and does not need this. */
    static float lnf_affine[T * M];
    for (int t = 0; t < T; t++) for (int m = 0; m < M; m++) lnf_affine[t * M + m] = g_lnf_out[t * M + m] * g_lnf_w[m] + g_lnf_b[m];

    int any_fail = 0;
    BoundaryStat s;
    s = compare_boundary("layer5", layer5_out, g_ref_layer5, T * M); any_fail |= s.fails > 0;
    if (!any_fail) { s = compare_boundary("layer11_final", layer11_out, g_ref_layer11, T * M); any_fail |= s.fails > 0; }
    if (!any_fail) { s = compare_boundary("ln_f", lnf_affine, g_ref_lnf, T * M); any_fail |= s.fails > 0; }
    if (!any_fail) { s = compare_boundary("logits", g_logits, g_ref_logits, T * VOCAB); any_fail |= s.fails > 0; }

    if (!any_fail) {
        int last = T - 1;
        int best[5] = {-1, -1, -1, -1, -1};
        float bestv[5] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (int v = 0; v < VOCAB; v++) {
            float val = g_logits[last * VOCAB + v];
            for (int r = 0; r < 5; r++) if (val > bestv[r]) {
                for (int q = 4; q > r; q--) { bestv[q] = bestv[q - 1]; best[q] = best[q - 1]; }
                bestv[r] = val; best[r] = v; break;
            }
        }
        printf("last-position top5 token ids: [%d,%d,%d,%d,%d] logits: [%.4f,%.4f,%.4f,%.4f,%.4f]\n",
               best[0], best[1], best[2], best[3], best[4], bestv[0], bestv[1], bestv[2], bestv[3], bestv[4]);
    }

    if (any_fail) { fprintf(stderr, "gpt2_full differential FAILED preregistered tolerance -- diagnostics stopped at first failing boundary\n"); return 1; }
    puts("gpt2_full differential check passed: all 12 real gpt2-124M blocks + ln_f + tied LM head, fingerprint-verified checkpoint read directly by this project's own safetensors loader, canonical graph+executor matches the PyTorch reference at layer5/layer11/ln_f/logits within preregistered tolerance");
    return 0;
}
