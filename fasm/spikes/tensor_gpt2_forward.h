#ifndef TENSOR_GPT2_FORWARD_H
#define TENSOR_GPT2_FORWARD_H
/* Runtime-sequence-length GPT-2 124M forward pass, for the generation loop.
 *
 * Why this exists separately from tensor_gpt2_full_differential_check.c's
 * canonical-graph pipeline: this project's canonical compiler bakes T (and
 * every other shape) into compile-time macros, so ONE compiled graph can
 * only ever execute at ONE fixed sequence length. A generation loop needs
 * a GROWING length at every step. This file mirrors the canonical graph's
 * exact per-op formulas (same math tensor_gpt2_full_differential_check.c
 * already anchored against the real canonical compile()+executor at
 * T=8 -- see that file) as plain, runtime-n-parameterized C functions --
 * the same "hand-rolled runtime reference, anchored once against the real
 * canonical op at a fixed shape" pattern this project already used for
 * the toy KV-cache and autoregressive-loop checks (fasm/spikes/
 * tensor_kv_cache_differential_check.c, tensor_autoregressive_loop_differential_check.c),
 * just now with the real GPT-2 124M weights and all 12 real blocks
 * instead of one toy block with random weights.
 *
 * Weight loading and the LayerNorm-affine-fold-into-the-following-matmul
 * trick are identical to tensor_gpt2_full_differential_check.c (see that
 * file's header comment for the exact linear-algebra identity and its
 * offline numerical verification) -- duplicated here rather than shared,
 * to avoid risking the already-passing, already-gated full-model
 * differential check while this generation-focused file is still being
 * developed and tested.
 */
#include "tensor_safetensors_loader.h"
#include "tensor_sha256.h"
#include <math.h>
#include <string.h>

#ifndef GPT2_M
#define GPT2_M 768
#endif
#ifndef GPT2_H
#define GPT2_H 12
#endif
#ifndef GPT2_D
#define GPT2_D 64
#endif
#ifndef GPT2_QW
#define GPT2_QW 2304
#endif
#ifndef GPT2_F
#define GPT2_F 3072
#endif
#ifndef GPT2_VOCAB
#define GPT2_VOCAB 50257
#endif
#ifndef GPT2_NLAYER
#define GPT2_NLAYER 12
#endif
#ifndef GPT2_MAXPOS
#define GPT2_MAXPOS 1024
#endif
#ifndef GPT2_MAXT
#define GPT2_MAXT 64
#endif

typedef struct {
    float ln1_w[GPT2_M], ln1_b[GPT2_M], ln2_w[GPT2_M], ln2_b[GPT2_M];
    float attn_w[GPT2_M * GPT2_QW], attn_b[GPT2_QW]; /* folded: ln1 affine baked in */
    float projw[GPT2_M * GPT2_M], projb[GPT2_M];
    float fc_w[GPT2_M * GPT2_F], fc_b[GPT2_F]; /* folded: ln2 affine baked in */
    float fcproj_w[GPT2_F * GPT2_M], fcproj_b[GPT2_M];
} Gpt2BlockWeights;

typedef struct {
    float wte[GPT2_VOCAB * GPT2_M];      /* embedding-table orientation, for lookup */
    float wpe[GPT2_MAXPOS * GPT2_M];
    float lnf_w_folded[GPT2_M * GPT2_VOCAB]; /* ln_f affine folded into the tied head */
    float lnf_b_folded[GPT2_VOCAB];
    Gpt2BlockWeights blk[GPT2_NLAYER];
} Gpt2Weights;

static void gpt2_fold_layernorm_affine(const float *gamma, const float *beta, float *w, float *b, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        double s = 0;
        for (int i = 0; i < in_dim; i++) s += (double)beta[i] * w[i * out_dim + o];
        b[o] += (float)s;
    }
    for (int i = 0; i < in_dim; i++) for (int o = 0; o < out_dim; o++) w[i * out_dim + o] *= gamma[i];
}

/* Loads real GPT-2 124M weights from a fingerprint-verified model.safetensors
 * (self-written SHA-256 + safetensors parser, no external library), folds
 * LayerNorm affines into the following matmul/tied-head weights. Returns 0
 * on success; nonzero (fail closed) on any missing/incompatible tensor or
 * fingerprint mismatch. expected_sha256 may be NULL to skip the check
 * (not recommended; callers should pass the pinned value). */
static int gpt2_load_weights(Gpt2Weights *w, const char *safetensors_path, const char *expected_sha256) {
    if (expected_sha256) {
        char actual[65];
        if (sha256_file(safetensors_path, actual)) return -1;
        if (strcmp(actual, expected_sha256)) {
            fprintf(stderr, "gpt2_load_weights: fingerprint mismatch: computed=%s expected=%s\n", actual, expected_sha256);
            return -1;
        }
    }
    SafetensorsFile st;
    if (st_open(&st, safetensors_path)) return -1;

    uint64_t s1[1] = {GPT2_M}, s2304[1] = {2304}, sMQW[2] = {GPT2_M, 2304}, sMM[2] = {GPT2_M, GPT2_M};
    uint64_t sF[1] = {GPT2_F}, sMF[2] = {GPT2_M, GPT2_F}, sFM[2] = {GPT2_F, GPT2_M};
    uint64_t sVM[2] = {GPT2_VOCAB, GPT2_M}, sPM[2] = {GPT2_MAXPOS, GPT2_M};
    int ok = 0;
    char name[64];
    for (int L = 0; L < GPT2_NLAYER; L++) {
        Gpt2BlockWeights *bw = &w->blk[L];
        float ln1_w[GPT2_M], ln1_b[GPT2_M], ln2_w[GPT2_M], ln2_b[GPT2_M];
        snprintf(name, sizeof name, "h.%d.ln_1.weight", L); ok |= st_read_f32(&st, name, s1, 1, ln1_w);
        snprintf(name, sizeof name, "h.%d.ln_1.bias", L); ok |= st_read_f32(&st, name, s1, 1, ln1_b);
        snprintf(name, sizeof name, "h.%d.ln_2.weight", L); ok |= st_read_f32(&st, name, s1, 1, ln2_w);
        snprintf(name, sizeof name, "h.%d.ln_2.bias", L); ok |= st_read_f32(&st, name, s1, 1, ln2_b);
        snprintf(name, sizeof name, "h.%d.attn.c_attn.weight", L); ok |= st_read_f32(&st, name, sMQW, 2, bw->attn_w);
        snprintf(name, sizeof name, "h.%d.attn.c_attn.bias", L); ok |= st_read_f32(&st, name, s2304, 1, bw->attn_b);
        snprintf(name, sizeof name, "h.%d.attn.c_proj.weight", L); ok |= st_read_f32(&st, name, sMM, 2, bw->projw);
        snprintf(name, sizeof name, "h.%d.attn.c_proj.bias", L); ok |= st_read_f32(&st, name, s1, 1, bw->projb);
        snprintf(name, sizeof name, "h.%d.mlp.c_fc.weight", L); ok |= st_read_f32(&st, name, sMF, 2, bw->fc_w);
        snprintf(name, sizeof name, "h.%d.mlp.c_fc.bias", L); ok |= st_read_f32(&st, name, sF, 1, bw->fc_b);
        snprintf(name, sizeof name, "h.%d.mlp.c_proj.weight", L); ok |= st_read_f32(&st, name, sFM, 2, bw->fcproj_w);
        snprintf(name, sizeof name, "h.%d.mlp.c_proj.bias", L); ok |= st_read_f32(&st, name, s1, 1, bw->fcproj_b);
        if (ok) { st_close(&st); return -1; }
        memcpy(bw->ln1_w, ln1_w, sizeof ln1_w); memcpy(bw->ln1_b, ln1_b, sizeof ln1_b);
        memcpy(bw->ln2_w, ln2_w, sizeof ln2_w); memcpy(bw->ln2_b, ln2_b, sizeof ln2_b);
        gpt2_fold_layernorm_affine(ln1_w, ln1_b, bw->attn_w, bw->attn_b, GPT2_M, GPT2_QW);
        gpt2_fold_layernorm_affine(ln2_w, ln2_b, bw->fc_w, bw->fc_b, GPT2_M, GPT2_F);
    }
    ok |= st_read_f32(&st, "wte.weight", sVM, 2, w->wte);
    ok |= st_read_f32(&st, "wpe.weight", sPM, 2, w->wpe);
    float lnf_w[GPT2_M], lnf_b[GPT2_M];
    ok |= st_read_f32(&st, "ln_f.weight", s1, 1, lnf_w);
    ok |= st_read_f32(&st, "ln_f.bias", s1, 1, lnf_b);
    st_close(&st);
    if (ok) return -1;

    for (int v = 0; v < GPT2_VOCAB; v++) for (int m = 0; m < GPT2_M; m++) w->lnf_w_folded[m * GPT2_VOCAB + v] = w->wte[v * GPT2_M + m];
    memset(w->lnf_b_folded, 0, sizeof w->lnf_b_folded);
    gpt2_fold_layernorm_affine(lnf_w, lnf_b, w->lnf_w_folded, w->lnf_b_folded, GPT2_M, GPT2_VOCAB);
    return 0;
}

static void gpt2_layernorm_raw(const float *x, int n, int dim, float *out) {
    for (int i = 0; i < n; i++) {
        double m = 0; for (int j = 0; j < dim; j++) m += x[i * dim + j]; m /= dim;
        double v = 0; for (int j = 0; j < dim; j++) { double d = x[i * dim + j] - m; v += d * d; } v /= dim;
        double invstd = 1.0 / sqrt(v + 1e-5);
        for (int j = 0; j < dim; j++) out[i * dim + j] = (float)((x[i * dim + j] - m) * invstd);
    }
}
static void gpt2_matmul_bias(const float *a, int n, int k, const float *w, const float *b, int outdim, float *out) {
    for (int i = 0; i < n; i++) for (int o = 0; o < outdim; o++) {
        float s = 0; for (int j = 0; j < k; j++) s += a[i * k + j] * w[j * outdim + o];
        out[i * outdim + o] = s + b[o];
    }
}
static void gpt2_gelu(const float *x, int count, float *out) {
    for (int i = 0; i < count; i++) {
        float v = x[i];
        float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        out[i] = 0.5f * v * (1.0f + tanhf(u));
    }
}
static void gpt2_causal_attention(const float *qkv, int n, float *out /* [n,GPT2_M] merged */) {
    float scale = 1.0f / sqrtf((float)GPT2_D);
    static float score[GPT2_MAXT];
    for (int h = 0; h < GPT2_H; h++) for (int i = 0; i < n; i++) {
        float mx = -INFINITY;
        for (int j = 0; j <= i; j++) {
            float s = 0;
            for (int d = 0; d < GPT2_D; d++) s += qkv[i * GPT2_QW + h * GPT2_D + d] * qkv[j * GPT2_QW + GPT2_M + h * GPT2_D + d];
            s *= scale; score[j] = s; if (s > mx) mx = s;
        }
        float total = 0, prob[GPT2_MAXT];
        for (int j = 0; j <= i; j++) { float e = expf(score[j] - mx); prob[j] = e; total += e; }
        for (int j = 0; j <= i; j++) prob[j] /= total;
        for (int d = 0; d < GPT2_D; d++) {
            float s = 0; for (int j = 0; j <= i; j++) s += prob[j] * qkv[j * GPT2_QW + 2 * GPT2_M + h * GPT2_D + d];
            out[i * GPT2_M + h * GPT2_D + d] = s;
        }
    }
}

/* Full runtime-n forward pass: token_ids[0..n) -> logits. n must be
 * <= GPT2_MAXT. If last_only, only the final position's logits are
 * written (to logits_out[0..GPT2_VOCAB), skipping the [n,VOCAB] tied-head
 * matmul's cost for the n-1 positions a generation loop never reads --
 * everything upstream (all 12 blocks) still processes the full context,
 * this only skips genuinely-unused output rows of the single largest
 * matmul. If !last_only, writes every position's logits, [n,GPT2_VOCAB]. */
static void gpt2_forward_ex(const Gpt2Weights *w, const int *token_ids, int n, float *logits_out, int last_only) {
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
    if (last_only) gpt2_matmul_bias(lnf + (n - 1) * GPT2_M, 1, GPT2_M, w->lnf_w_folded, w->lnf_b_folded, GPT2_VOCAB, logits_out);
    else gpt2_matmul_bias(lnf, n, GPT2_M, w->lnf_w_folded, w->lnf_b_folded, GPT2_VOCAB, logits_out);
}

static void gpt2_forward(const Gpt2Weights *w, const int *token_ids, int n, float *logits_out) {
    gpt2_forward_ex(w, token_ids, n, logits_out, 0);
}

#endif
