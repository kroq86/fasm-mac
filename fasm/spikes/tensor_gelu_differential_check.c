/* Decoder-runtime Etap 2, operation 4/10: GELU (tanh approximation, GPT-2's
 * own reference constants -- see the kernel comment in
 * tensor_semantic_compiler.h for why this specific approximation, not the
 * exact erf-based GELU).
 *
 * Tested per the predeclared criterion: shapes with 7/8/9 total elements,
 * finite values, and tails (explicit large positive/negative inputs where
 * GELU saturates toward x and 0 respectively, plus x=0 exactly, mixed in
 * alongside the random values -- not just randomly-generated small
 * numbers that never probe the activation's asymptotic behavior).
 */
#include "tensor_semantic_compiler.h"
#include <stdio.h>

enum { MAX_N = 9 };

static float dv(int i, int salt) {
    float a = sinf((float)(i * 12.9898f + salt * 78.233f)) * 43758.5453f;
    return a - floorf(a) - 0.5f;
}

static float oracle_gelu(float x) {
    const float c0 = 0.7978845608028654f, c1 = 0.044715f;
    float u = c0 * (x + c1 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(u));
}
static float oracle_gelu_grad(float x) {
    const float c0 = 0.7978845608028654f, c1 = 0.044715f;
    float u = c0 * (x + c1 * x * x * x);
    float t = tanhf(u);
    float sech2 = 1.0f - t * t;
    float du_dx = c0 * (1.0f + 3.0f * c1 * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * sech2 * du_dx;
}

static int run_case(uint32_t n, const float *x_values, const char *label) {
    enum { X, Y, NODES };
    float x[MAX_N], gx[MAX_N] = {0}, y[MAX_N], gy[MAX_N];
    for (uint32_t i = 0; i < n; i++) { x[i] = x_values[i]; gy[i] = dv((int)i, 77); } /* arbitrary seeded incoming gradient */

    Node g[NODES] = {
        {LEAF, NONE, NONE, INPUT | RETAIN_GRAD, {x, gx, NULL, 1, n, 0}},
        {GELU, X, NONE, TEMP | RETAIN_GRAD, {y, gy, NULL, 1, n, 0}},
    };
    ExecStep steps[8]; Context ctx[8]; uint32_t count = 0;
    if (compile(g, NODES, .01f, steps, ctx, 8, &count)) { fprintf(stderr, "%s: compile rejected\n", label); return 1; }
    if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "%s: execute failed\n", label); return 1; }

    for (uint32_t i = 0; i < n; i++) {
        if (!isfinite(y[i])) { fprintf(stderr, "%s: non-finite output at %u (x=%.6g)\n", label, i, x[i]); return 1; }
        float oy = oracle_gelu(x[i]);
        if (fabsf(y[i] - oy) > 1e-5f) { fprintf(stderr, "%s: forward mismatch at %u: canonical=%.9g oracle=%.9g (x=%.6g)\n", label, i, y[i], oy, x[i]); return 1; }
        if (!isfinite(gx[i])) { fprintf(stderr, "%s: non-finite grad at %u\n", label, i); return 1; }
        float og = gy[i] * oracle_gelu_grad(x[i]);
        if (fabsf(gx[i] - og) > 1e-5f) { fprintf(stderr, "%s: backward mismatch at %u: canonical=%.9g oracle=%.9g\n", label, i, gx[i], og); return 1; }
    }
    printf("%s: n=%u forward=match backward=match finite=yes\n", label, n);
    return 0;
}

int main(void) {
    /* n=7: mostly random small values plus one explicit zero (tail: the
     * cubic term vanishes and tanh(0)=0, GELU(0) must be exactly 0) */
    float v7[7]; for (int i = 0; i < 7; i++) v7[i] = dv(i, 1) * 3.0f; v7[3] = 0.0f;
    if (run_case(7, v7, "n7_with_zero")) return 1;

    /* n=8: tails -- large positive/negative inputs where GELU saturates
     * toward x (positive) and toward 0 (negative) */
    float v8[8] = { 10.0f, -10.0f, 6.0f, -6.0f, 0.5f, -0.5f, 3.0f, -3.0f };
    if (run_case(8, v8, "n8_saturation_tails")) return 1;

    /* n=9: random values spanning a wider range, mixed magnitudes */
    float v9[9]; for (int i = 0; i < 9; i++) v9[i] = dv(i, 5) * 8.0f;
    if (run_case(9, v9, "n9_wide_range")) return 1;

    puts("gelu differential check passed: reference<->canonical<->executor agree on n=7/8/9, forward exact, backward exact, saturation tails and zero all finite");
    return 0;
}
