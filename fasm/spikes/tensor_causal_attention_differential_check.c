/* Decoder-runtime Etap 2, operation 3/10: causal attention mask.
 *
 * T is a build-time override (-DT=7/8/9), following this header's own
 * #ifndef-guarded shape-constant convention -- one source file, the gate
 * script compiles it three times for the three required shapes rather
 * than needing three separate .c files or a runtime-shaped op.
 *
 * Tested per the predeclared criterion:
 *   - finite values;
 *   - causal isolation: prob[h,i,j] for every j>i is EXACTLY 0.0, not
 *     just small -- verified directly against the raw aux buffer, not
 *     inferred from the output;
 *   - forward matches an independent oracle causal-attention
 *     implementation (hand-written separately, not sharing code with
 *     k_causal_attention_fwd);
 *   - backward matches central finite differences on qkv (not a second
 *     hand-derived analytic backward -- re-deriving the same softmax
 *     backward algebra a second time risks the same mistake twice;
 *     finite differences are an independent numerical method, the same
 *     convention tensor_transformer_reference_spike.h's own check()
 *     already uses in this project).
 */
#ifndef T
#define T 8
#endif
#define M 4
#define H 2
#define D 2
#define QW (3*M)
#include "tensor_semantic_compiler.h"
#include <stdio.h>
#include <stdlib.h>

static float dv(int i, int salt) {
    float a = sinf((float)(i * 12.9898f + salt * 78.233f)) * 43758.5453f;
    return a - floorf(a) - 0.5f;
}

/* Independent oracle: same mathematical definition, written without
 * reusing k_causal_attention_fwd's code. */
static void oracle_causal_attention(const float *qkv, float *out, float *prob) {
    float scale = 1.0f / sqrtf((float)D);
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
        float score[T], mx = -1e30f;
        for (int j = 0; j <= i; j++) {
            float s = 0;
            for (int d = 0; d < D; d++) s += qkv[i * QW + h * D + d] * qkv[j * QW + M + h * D + d];
            score[j] = s * scale;
            if (score[j] > mx) mx = score[j];
        }
        float total = 0;
        for (int j = 0; j <= i; j++) { score[j] = expf(score[j] - mx); total += score[j]; }
        for (int j = 0; j < T; j++) prob[(h * T + i) * T + j] = j <= i ? score[j] / total : 0.0f;
        for (int d = 0; d < D; d++) {
            float s = 0;
            for (int j = 0; j <= i; j++) s += prob[(h * T + i) * T + j] * qkv[j * QW + 2 * M + h * D + d];
            out[(h * T + i) * D + d] = s;
        }
    }
}

/* Runs forward through the canonical graph for the given qkv buffer,
 * returns the scalar sum of the attention output (used as the loss for
 * finite-difference gradient checking). */
static float canonical_forward_sum(float *qkv, float *out, float *prob) {
    enum { QKVN, ATT, NODES };
    float gqkv[T * QW] = {0}, gout[H * T * D];
    Node g[NODES] = {
        {LEAF, NONE, NONE, INPUT | RETAIN_GRAD, {qkv, gqkv, NULL, T, QW, 0}},
        {CAUSAL_ATTENTION, QKVN, NONE, TEMP | RETAIN_GRAD, {out, gout, prob, 1, H * T * D, H * T * T}},
    };
    ExecStep steps[8]; Context ctx[8]; uint32_t count = 0;
    if (compile(g, NODES, .01f, steps, ctx, 8, &count)) { fprintf(stderr, "compile rejected\n"); exit(1); }
    if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "execute failed\n"); exit(1); }
    float sum = 0; for (int i = 0; i < H * T * D; i++) sum += out[i];
    return sum;
}

int main(void) {
    float qkv[T * QW];
    for (int i = 0; i < T * QW; i++) qkv[i] = dv(i, 42) * 0.6f;

    float out[H * T * D], prob[H * T * T];
    float oracle_out[H * T * D], oracle_prob[H * T * T];
    canonical_forward_sum(qkv, out, prob);
    oracle_causal_attention(qkv, oracle_out, oracle_prob);

    for (int i = 0; i < H * T * D; i++) {
        if (!isfinite(out[i])) { fprintf(stderr, "non-finite output at %d\n", i); return 1; }
        if (fabsf(out[i] - oracle_out[i]) > 1e-5f) { fprintf(stderr, "forward mismatch at %d: canonical=%.9g oracle=%.9g\n", i, out[i], oracle_out[i]); return 1; }
    }

    /* causal isolation: prob[h,i,j] for j>i must be EXACTLY 0.0 */
    int leaked = 0;
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) for (int j = i + 1; j < T; j++)
        if (prob[(h * T + i) * T + j] != 0.0f) leaked++;
    if (leaked) { fprintf(stderr, "causal isolation violated: %d future-position weights are nonzero\n", leaked); return 1; }

    /* each row's causal probabilities must still sum to 1 (softmax normalization intact under masking) */
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
        float s = 0; for (int j = 0; j <= i; j++) s += prob[(h * T + i) * T + j];
        if (fabsf(s - 1.0f) > 1e-5f) { fprintf(stderr, "row h=%d i=%d causal probabilities do not sum to 1 (sum=%.9g)\n", h, i, s); return 1; }
    }

    /* backward vs central finite differences on a spread of qkv entries
     * (query, key, and value slots, early/mid/late T positions) */
    int check_idx[] = { 0, M, 2 * M, (T / 2) * QW, (T / 2) * QW + M, (T - 1) * QW, (T - 1) * QW + 2 * M + D };
    float eps = 1e-3f;
    for (unsigned k = 0; k < sizeof check_idx / sizeof check_idx[0]; k++) {
        int idx = check_idx[k];
        float saved = qkv[idx];
        float dummy_out[H * T * D], dummy_prob[H * T * T];
        qkv[idx] = saved + eps; float plus = canonical_forward_sum(qkv, dummy_out, dummy_prob);
        qkv[idx] = saved - eps; float minus = canonical_forward_sum(qkv, dummy_out, dummy_prob);
        qkv[idx] = saved;
        float numeric = (plus - minus) / (2 * eps);

        /* analytic gradient: seed d(sum)/d(out)=1 everywhere, run backward */
        enum { QKVN, ATT, NODES };
        float gqkv[T * QW] = {0}, gout[H * T * D];
        for (int i = 0; i < H * T * D; i++) gout[i] = 1.0f;
        Node g[NODES] = {
            {LEAF, NONE, NONE, INPUT | RETAIN_GRAD, {qkv, gqkv, NULL, T, QW, 0}},
            {CAUSAL_ATTENTION, QKVN, NONE, TEMP | RETAIN_GRAD, {dummy_out, gout, dummy_prob, 1, H * T * D, H * T * T}},
        };
        ExecStep steps[8]; Context ctx[8]; uint32_t count = 0;
        if (compile(g, NODES, .01f, steps, ctx, 8, &count)) return 1;
        /* forward (to populate prob) then backward only, using the pre-seeded gout above */
        if (tensor_transformer_steps_execute(steps, 1)) return 1; /* the single FORWARD step */
        if (tensor_transformer_steps_execute(steps + count - 1, 1)) return 1; /* the single BACKWARD step (ZERO_GRAD is skipped; gqkv was fresh) */
        float analytic = gqkv[idx];
        float tol = 5e-3f * fmaxf(1.0f, fmaxf(fabsf(numeric), fabsf(analytic)));
        if (fabsf(numeric - analytic) > tol) { fprintf(stderr, "backward mismatch at qkv[%d]: analytic=%.9g numeric=%.9g\n", idx, analytic, numeric); return 1; }
    }

    printf("causal_attention differential check passed: T=%d forward=match causal_isolation=exact-zero row_sums=1 backward=finite-diff-match\n", T);
    return 0;
}
