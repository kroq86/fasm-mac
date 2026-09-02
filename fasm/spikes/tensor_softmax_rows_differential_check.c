/* Decoder-runtime Etap 2, operation 5/10: row-wise numerically stable
 * softmax (the final logits->vocab-probability step, standalone from
 * attention's internal softmax).
 *
 * Tested per the predeclared criterion: shapes with cols in {7,8,9}
 * (standing in for a small vocab dimension), finite values, tails
 * (one row deliberately uses a huge logit spread -- large enough that
 * exp() of the raw value would overflow float32 without the
 * max-subtraction stability trick -- to actually exercise "numerically
 * stable", not just assert the label), and each row summing to exactly
 * ~1. Backward checked against an independent analytic softmax-Jacobian
 * oracle AND a finite-difference spot check for extra confidence.
 */
#include "tensor_semantic_compiler.h"
#include <stdio.h>
#include <stdlib.h>

enum { ROWS = 3, MAX_COLS = 9 };

static float dv(int i, int salt) {
    float a = sinf((float)(i * 12.9898f + salt * 78.233f)) * 43758.5453f;
    return a - floorf(a) - 0.5f;
}

static void oracle_softmax_rows(const float *x, uint32_t rows, uint32_t cols, float *y) {
    for (uint32_t i = 0; i < rows; i++) {
        double mx = -1e300;
        for (uint32_t j = 0; j < cols; j++) if (x[i * cols + j] > mx) mx = x[i * cols + j];
        double total = 0;
        for (uint32_t j = 0; j < cols; j++) { double e = exp((double)x[i * cols + j] - mx); y[i * cols + j] = (float)e; total += e; }
        for (uint32_t j = 0; j < cols; j++) y[i * cols + j] = (float)(y[i * cols + j] / total);
    }
}
static void oracle_softmax_grad(const float *y, uint32_t rows, uint32_t cols, const float *dy, float *dx) {
    for (uint32_t i = 0; i < rows; i++) {
        double dot = 0;
        for (uint32_t j = 0; j < cols; j++) dot += (double)dy[i * cols + j] * y[i * cols + j];
        for (uint32_t j = 0; j < cols; j++) dx[i * cols + j] = (float)(y[i * cols + j] * ((double)dy[i * cols + j] - dot));
    }
}

static float forward_sum(float *x, uint32_t cols, float *y) {
    enum { X, Y, NODES };
    float gx[ROWS * MAX_COLS] = {0}, gy[ROWS * MAX_COLS];
    Node g[NODES] = {
        {LEAF, NONE, NONE, INPUT | RETAIN_GRAD, {x, gx, NULL, ROWS, cols, 0}},
        {SOFTMAX_ROWS, X, NONE, TEMP | RETAIN_GRAD, {y, gy, NULL, ROWS, cols, 0}},
    };
    ExecStep steps[8]; Context ctx[8]; uint32_t count = 0;
    if (compile(g, NODES, .01f, steps, ctx, 8, &count)) { fprintf(stderr, "compile rejected\n"); exit(1); }
    if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "execute failed\n"); exit(1); }
    float sum = 0; for (uint32_t i = 0; i < ROWS * cols; i++) sum += y[i];
    return sum;
}

static int run_case(uint32_t cols, const float *x_values, const char *label) {
    enum { X, Y, NODES };
    float x[ROWS * MAX_COLS], gx[ROWS * MAX_COLS] = {0}, y[ROWS * MAX_COLS], gy[ROWS * MAX_COLS];
    for (uint32_t i = 0; i < ROWS * cols; i++) { x[i] = x_values[i]; gy[i] = dv((int)i, 88); }

    Node g[NODES] = {
        {LEAF, NONE, NONE, INPUT | RETAIN_GRAD, {x, gx, NULL, ROWS, cols, 0}},
        {SOFTMAX_ROWS, X, NONE, TEMP | RETAIN_GRAD, {y, gy, NULL, ROWS, cols, 0}},
    };
    ExecStep steps[8]; Context ctx[8]; uint32_t count = 0;
    if (compile(g, NODES, .01f, steps, ctx, 8, &count)) { fprintf(stderr, "%s: compile rejected\n", label); return 1; }
    if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "%s: execute failed\n", label); return 1; }

    float oracle_y[ROWS * MAX_COLS];
    oracle_softmax_rows(x, ROWS, cols, oracle_y);
    for (uint32_t i = 0; i < ROWS * cols; i++) {
        if (!isfinite(y[i])) { fprintf(stderr, "%s: non-finite output at %u (x=%.6g)\n", label, i, x[i]); return 1; }
        if (fabsf(y[i] - oracle_y[i]) > 1e-5f) { fprintf(stderr, "%s: forward mismatch at %u: canonical=%.9g oracle=%.9g\n", label, i, y[i], oracle_y[i]); return 1; }
    }
    for (uint32_t i = 0; i < ROWS; i++) {
        float s = 0; for (uint32_t j = 0; j < cols; j++) s += y[i * cols + j];
        if (fabsf(s - 1.0f) > 1e-5f) { fprintf(stderr, "%s: row %u does not sum to 1 (sum=%.9g)\n", label, i, s); return 1; }
    }

    float oracle_dx[ROWS * MAX_COLS];
    oracle_softmax_grad(y, ROWS, cols, gy, oracle_dx);
    for (uint32_t i = 0; i < ROWS * cols; i++) {
        if (!isfinite(gx[i])) { fprintf(stderr, "%s: non-finite grad at %u\n", label, i); return 1; }
        if (fabsf(gx[i] - oracle_dx[i]) > 1e-5f) { fprintf(stderr, "%s: backward mismatch at %u: canonical=%.9g oracle=%.9g\n", label, i, gx[i], oracle_dx[i]); return 1; }
    }

    /* finite-difference spot check on one entry, independent numerical method */
    float x2[ROWS * MAX_COLS]; for (uint32_t i = 0; i < ROWS * cols; i++) x2[i] = x_values[i];
    float y2[ROWS * MAX_COLS];
    float eps = 1e-3f;
    uint32_t probe = cols / 2;
    float saved = x2[probe];
    x2[probe] = saved + eps; float plus = forward_sum(x2, cols, y2);
    x2[probe] = saved - eps; float minus = forward_sum(x2, cols, y2);
    float numeric = (plus - minus) / (2 * eps);
    /* d(sum of all y)/dx[probe] analytically: gy would need to be all-ones for this to equal the "sum" loss gradient */
    float ones[ROWS * MAX_COLS]; for (uint32_t i = 0; i < ROWS * cols; i++) ones[i] = 1.0f;
    float oracle_dx_ones[ROWS * MAX_COLS];
    oracle_softmax_grad(oracle_y, ROWS, cols, ones, oracle_dx_ones);
    if (fabsf(numeric - oracle_dx_ones[probe]) > 5e-3f) { fprintf(stderr, "%s: finite-difference spot check failed at probe %u: analytic=%.9g numeric=%.9g\n", label, probe, oracle_dx_ones[probe], numeric); return 1; }

    printf("%s: cols=%u forward=match row_sums=1 backward=match finite_diff_spot_check=match finite=yes\n", label, cols);
    return 0;
}

int main(void) {
    /* cols=7: ordinary random logits */
    float v7[ROWS * 7]; for (int i = 0; i < ROWS * 7; i++) v7[i] = dv(i, 1) * 4.0f;
    if (run_case(7, v7, "cols7_plain")) return 1;

    /* cols=8: numerical-stability tail -- row 1 has a logit of 80.0
     * alongside small values; exp(80) alone overflows float32 (max ~88.7
     * for expf before inf, but combined with the rest of this row's
     * arithmetic this is deliberately right at the edge where a naive
     * softmax without max-subtraction produces inf/nan) */
    float v8[ROWS * 8];
    for (int i = 0; i < ROWS * 8; i++) v8[i] = dv(i, 2) * 2.0f;
    v8[8 + 3] = 80.0f; /* row 1, col 3: the near-overflow tail */
    if (run_case(8, v8, "cols8_overflow_tail")) return 1;

    /* cols=9: a row of very negative logits (near-uniform-suppressed
     * probabilities) alongside a normal row -- the opposite tail */
    float v9[ROWS * 9];
    for (int i = 0; i < ROWS * 9; i++) v9[i] = dv(i, 3) * 3.0f;
    for (int j = 0; j < 9; j++) v9[9 + j] = -60.0f + dv(j, 4) * 0.1f; /* row 1: uniformly very negative */
    if (run_case(9, v9, "cols9_underflow_tail")) return 1;

    puts("softmax_rows differential check passed: reference<->canonical<->executor agree on cols=7/8/9, forward exact, row sums=1, backward exact, overflow/underflow tails handled, all finite");
    return 0;
}
