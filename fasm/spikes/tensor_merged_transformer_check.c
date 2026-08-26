/* Merged-compiler regression check: Transformer, via the single canonical
 * compile(). Same real Block shape as every other transformer spike in
 * this repo (T=3,M=4,H=2,D=2,F=6,QW=3*M). This file is ONLY a graph +
 * tensors (criterion: no forward_nodes[]/zeroable[]/hand-written schedule
 * code per model) — it must reproduce the exact behavior
 * tensor_graph_engine_transformer_check.c already reached (finite
 * differences on wq/wo/w1/w2, loss collapsing well below its initial
 * value), as a regression check that the merge didn't change anything.
 * ATTENTION and LAYERNORM go through the generic `aux` saved-state field
 * on Tensor (criterion 7) — no transformer-special struct fields exist
 * anywhere in tensor_semantic_compiler.h.
 */
#define T 3
#define M 4
#define H 2
#define D 2
#define F 6
#define QW (3 * M)
#include "tensor_semantic_compiler.h"
#include <stdio.h>
#include <string.h>

static float initv(int i, int s) { return (float)(((i * 37 + s * 17) % 29) - 14) / 41.0f; }

enum {
    N_X, N_WQ, N_WO, N_W1, N_W2, N_TARGET,
    N_QKV, N_ATTN, N_MERGED, N_PROJ, N_SUM1, N_LN1, N_Z1, N_ACT, N_FF, N_SUM2, N_LN2, N_LOSS,
    N_COUNT
};

int main(void) {
    float x[T * M], wq[M * QW], wo[M * M], w1[M * F], w2[F * M], target[T * M];
    float qkv[T * QW], attn[H * T * D], merged[T * M], proj[T * M], sum1[T * M], ln1[T * M];
    float z1[T * F], act[T * F], ff[T * M], sum2[T * M], ln2[T * M], loss[1];
    float gx[T * M] = {0}, gwq[M * QW], gwo[M * M], gw1[M * F], gw2[F * M], gtarget[T * M] = {0};
    float gqkv[T * QW], gattn[H * T * D], gmerged[T * M], gproj[T * M], gsum1[T * M], gln1[T * M];
    float gz1[T * F], gact[T * F], gff[T * M], gsum2[T * M], gln2[T * M], gloss[1];
    float attn_aux[H * T * T], ln1_aux[2 * T], ln2_aux[2 * T];

    for (int i = 0; i < T * M; i++) x[i] = initv(i, 1);
    for (int i = 0; i < M * QW; i++) wq[i] = initv(i, 2) * .4f;
    for (int i = 0; i < M * M; i++) wo[i] = initv(i, 3) * .4f;
    for (int i = 0; i < M * F; i++) w1[i] = initv(i, 4) * .5f;
    for (int i = 0; i < F * M; i++) w2[i] = initv(i, 5) * .5f;
    for (int i = 0; i < T; i++) {
        float m = 0, v = 0;
        for (int j = 0; j < M; j++) { target[i * M + j] = sinf((float)((i + 1) * (j + 2)) * .7f) + cosf((float)(i - j) * .4f); m += target[i * M + j]; }
        m /= M;
        for (int j = 0; j < M; j++) { float d = target[i * M + j] - m; v += d * d; }
        float inv = 1.0f / sqrtf(v / M + 1e-5f);
        for (int j = 0; j < M; j++) target[i * M + j] = (target[i * M + j] - m) * inv;
    }

    Node g[N_COUNT] = {
        [N_X] = {LEAF, NONE, NONE, INPUT, {x, gx, NULL, T, M, 0}},
        [N_WQ] = {LEAF, NONE, NONE, PARAM, {wq, gwq, NULL, M, QW, 0}},
        [N_WO] = {LEAF, NONE, NONE, PARAM, {wo, gwo, NULL, M, M, 0}},
        [N_W1] = {LEAF, NONE, NONE, PARAM, {w1, gw1, NULL, M, F, 0}},
        [N_W2] = {LEAF, NONE, NONE, PARAM, {w2, gw2, NULL, F, M, 0}},
        [N_TARGET] = {LEAF, NONE, NONE, CONSTANT, {target, gtarget, NULL, T, M, 0}},
        [N_QKV] = {MATMUL, N_X, N_WQ, TEMP, {qkv, gqkv, NULL, T, QW, 0}},
        [N_ATTN] = {ATTENTION, N_QKV, NONE, TEMP, {attn, gattn, attn_aux, 1, H * T * D, H * T * T}},
        [N_MERGED] = {CONTIGUOUS, N_ATTN, NONE, TEMP, {merged, gmerged, NULL, T, M, 0}},
        [N_PROJ] = {MATMUL, N_MERGED, N_WO, TEMP, {proj, gproj, NULL, T, M, 0}},
        [N_SUM1] = {RESIDUAL, N_X, N_PROJ, TEMP, {sum1, gsum1, NULL, T, M, 0}},
        [N_LN1] = {LAYERNORM, N_SUM1, NONE, TEMP, {ln1, gln1, ln1_aux, T, M, 2 * T}},
        [N_Z1] = {MATMUL, N_LN1, N_W1, TEMP, {z1, gz1, NULL, T, F, 0}},
        [N_ACT] = {RELU, N_Z1, NONE, TEMP, {act, gact, NULL, T, F, 0}},
        [N_FF] = {MATMUL, N_ACT, N_W2, TEMP, {ff, gff, NULL, T, M, 0}},
        [N_SUM2] = {RESIDUAL, N_LN1, N_FF, TEMP, {sum2, gsum2, NULL, T, M, 0}},
        [N_LN2] = {LAYERNORM, N_SUM2, NONE, TEMP, {ln2, gln2, ln2_aux, T, M, 2 * T}},
        [N_LOSS] = {MSE, N_LN2, N_TARGET, TEMP, {loss, gloss, NULL, 1, 1, 0}},
    };

    ExecStep steps[64];
    Context ctx[64];
    uint32_t count = 0;
    if (compile(g, N_COUNT, 0.02f, steps, ctx, 64, &count)) { fprintf(stderr, "compile failed\n"); return 1; }
    /* forward=12 zero=15 (4 params + 11 non-MSE temps) backward=12 (incl.
     * MSE) optimizer=4 -> 43, all compile()-derived */
    if (count != 43) { fprintf(stderr, "unexpected step count=%u (want 43)\n", count); return 1; }

    Group groups[N_COUNT];
    unsigned ng = detect_fusion(g, N_COUNT, groups);
    int all_plain = 1;
    for (unsigned i = 0; i < ng; i++) if (groups[i].kind != GROUP_PLAIN) all_plain = 0;
    if (ng != 12 || !all_plain) {
        fprintf(stderr, "fusion pass found unexpected groups on the no-bias transformer graph: ng=%u all_plain=%d\n", ng, all_plain);
        return 8;
    }
    printf("fusion cross-check: same generic pass as the MLP graph, applied unmodified -> groups=%u all_plain=yes\n", ng);

    /* correctness: finite differences, probing one weight from each of the
     * four trainable groups. steps[] layout from compile(): forward(12)
     * zero(15) backward(12) optimizer(4) — run only forward(0..12) to
     * re-check loss at a perturbed weight so the optimizer slice never
     * touches params mid-check. */
    if (tensor_transformer_steps_execute(steps, 12 + 15 + 12)) return 2; /* forward+zero+backward, no optimizer */
    struct { float *param, *grad; int idx; const char *name; } probes[] = {
        {wq, gwq, 5, "wq"}, {wo, gwo, 3, "wo"}, {w1, gw1, 7, "w1"}, {w2, gw2, 2, "w2"},
    };
    for (unsigned p = 0; p < 4; p++) {
        float eps = 1e-3f, old = probes[p].param[probes[p].idx], analytic = probes[p].grad[probes[p].idx];
        probes[p].param[probes[p].idx] = old + eps;
        if (tensor_transformer_steps_execute(steps, 12)) return 2;
        float plus = loss[0];
        probes[p].param[probes[p].idx] = old - eps;
        if (tensor_transformer_steps_execute(steps, 12)) return 2;
        float minus = loss[0];
        probes[p].param[probes[p].idx] = old;
        float numeric = (plus - minus) / (2 * eps);
        if (fabsf(numeric - analytic) > 5e-3f * fmaxf(1, fmaxf(fabsf(numeric), fabsf(analytic)))) {
            fprintf(stderr, "merged compiler transformer backward mismatch on %s: analytic=%g numeric=%g\n", probes[p].name, analytic, numeric);
            return 3;
        }
    }

    for (int i = 0; i < M * QW; i++) wq[i] = initv(i, 2) * .4f;
    for (int i = 0; i < M * M; i++) wo[i] = initv(i, 3) * .4f;
    for (int i = 0; i < M * F; i++) w1[i] = initv(i, 4) * .5f;
    for (int i = 0; i < F * M; i++) w2[i] = initv(i, 5) * .5f;
    if (tensor_transformer_steps_execute(steps, 12)) return 4;
    float initial = loss[0];
    for (unsigned epoch = 0; epoch < 12000; epoch++) if (tensor_transformer_steps_execute(steps, count)) return 4;
    if (tensor_transformer_steps_execute(steps, 12)) return 4;
    float final = loss[0];

    printf("merged compiler check (transformer) passed: nodes=%u steps=%u (forward=12 zero=15 backward=12 optimizer=4, "
           "all compile()-derived, no forward_nodes[]/zeroable[] in this file) ops=matmul,attention,contiguous,residual,layernorm,relu,mse "
           "aux_used_by=attention,layernorm (generic saved-state, no transformer-special Node fields) "
           "gradient_check=finite-difference(wq,wo,w1,w2) epochs=12000 loss=%.6f->%.6f\n",
           (unsigned)N_COUNT, count, initial, final);
    return final < initial * .25f ? 0 : 6;
}
