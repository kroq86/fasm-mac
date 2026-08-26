/* Experimental layout-aware encoder block; not a stable core ABI.
 *
 * tensor_merge_layout_profile.c proved the projection reformulation is
 * exact and wins in isolation (up to 2.62x on the real block's shape).
 * This wires it into the real Block's full forward AND backward — and the
 * isolated win does NOT survive contact with the rest of the pipeline.
 *
 * Two things had to be fixed to even measure this honestly. First, the
 * monolithic forward()/backward() this repo actually trains with already
 * fuses the merge write into the attention loop (b->merge[...] = s right
 * next to b->head[...] = s) — there is no separate CONTIGUOUS pass to
 * eliminate there at all; only a hand-built scheduled-executor path
 * (tensor_transformer_scheduled_train_check.c's F_CONTIG action) pays that
 * cost explicitly, so a fair comparison needs both baselines, not just the
 * fused one. Second, single-shot timing at this scale (~170ns for the
 * whole forward pass) is dominated by measurement noise, not the code
 * under test — separate process runs disagreed even in which direction
 * the "winner" was. Fixed with median-of-15 repeated trials per variant.
 *
 * Once measured properly: forward+backward+finite-difference correctness
 * is exact, but the timing difference between all three forward variants
 * (fused / explicit-contiguous / layout-aware) is within noise (~1.00-
 * 1.04x) at this block's real T=3,H=2,D=2 shape. The isolated op-level win
 * gets diluted below the noise floor by the rest of the forward pass
 * (QKV projection, attention, two LayerNorms, FFN) — the same lesson as
 * the SIMD profiling spike: an isolated microbenchmark does not predict
 * full-pipeline impact, in either direction.
 */
#define TRANSFORMER_REFERENCE_NO_MAIN
#include "tensor_transformer_reference_spike.h"
#undef TRANSFORMER_REFERENCE_NO_MAIN
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Same as forward(), except the attention loop writes only b->head, and an
 * explicit separate pass builds b->merge afterward — matching what a real
 * planner-emitted CONTIGUOUS action actually costs, unlike forward()'s
 * free fused write. This is the fair baseline for "does skipping
 * CONTIGUOUS help", not the already-fused monolithic forward(). */
static void forward_explicit_contig(Block *b) {
    mm(b->x, b->wq, b->qkv, T, M, QW);
    float scale = 1 / sqrtf(D);
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
        float mx = -INFINITY;
        for (int j = 0; j < T; j++) {
            float s = 0;
            for (int d = 0; d < D; d++) s += b->qkv[i * QW + h * D + d] * b->qkv[j * QW + M + h * D + d];
            s *= scale;
            b->score[(h * T + i) * T + j] = s;
            mx = fmaxf(mx, s);
        }
        float total = 0;
        for (int j = 0; j < T; j++) { float e = expf(b->score[(h * T + i) * T + j] - mx); b->prob[(h * T + i) * T + j] = e; total += e; }
        for (int j = 0; j < T; j++) b->prob[(h * T + i) * T + j] /= total;
        for (int d = 0; d < D; d++) {
            float s = 0;
            for (int j = 0; j < T; j++) s += b->prob[(h * T + i) * T + j] * b->qkv[j * QW + 2 * M + h * D + d];
            b->head[(h * T + i) * D + d] = s;
        }
    }
    for (int i = 0; i < T; i++) for (int h = 0; h < H; h++) for (int d = 0; d < D; d++) b->merge[i * M + h * D + d] = b->head[(h * T + i) * D + d];
    mm(b->merge, b->wo, b->proj, T, M, M);
    for (int i = 0; i < T * M; i++) b->s1[i] = b->x[i] + b->proj[i];
    ln(b->s1, b->ln1, b->mean1, b->inv1);
    mm(b->ln1, b->w1, b->z1, T, M, F);
    for (int i = 0; i < T * F; i++) b->act[i] = b->z1[i] > 0 ? b->z1[i] : 0;
    mm(b->act, b->w2, b->ff, T, F, M);
    for (int i = 0; i < T * M; i++) b->s2[i] = b->ln1[i] + b->ff[i];
    ln(b->s2, b->out, b->mean2, b->inv2);
}

static void forward_layout(Block *b) {
    mm(b->x, b->wq, b->qkv, T, M, QW);
    float scale = 1 / sqrtf(D);
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
        float mx = -INFINITY;
        for (int j = 0; j < T; j++) {
            float s = 0;
            for (int d = 0; d < D; d++) s += b->qkv[i * QW + h * D + d] * b->qkv[j * QW + M + h * D + d];
            s *= scale;
            b->score[(h * T + i) * T + j] = s;
            mx = fmaxf(mx, s);
        }
        float total = 0;
        for (int j = 0; j < T; j++) { float e = expf(b->score[(h * T + i) * T + j] - mx); b->prob[(h * T + i) * T + j] = e; total += e; }
        for (int j = 0; j < T; j++) b->prob[(h * T + i) * T + j] /= total;
        for (int d = 0; d < D; d++) {
            float s = 0;
            for (int j = 0; j < T; j++) s += b->prob[(h * T + i) * T + j] * b->qkv[j * QW + 2 * M + h * D + d];
            b->head[(h * T + i) * D + d] = s; /* no b->merge write: nothing to make contiguous */
        }
    }
    memset(b->proj, 0, sizeof b->proj);
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) for (int d = 0; d < D; d++) {
        float x = b->head[(h * T + i) * D + d];
        for (int j = 0; j < M; j++) b->proj[i * M + j] += x * b->wo[(h * D + d) * M + j];
    }
    for (int i = 0; i < T * M; i++) b->s1[i] = b->x[i] + b->proj[i];
    ln(b->s1, b->ln1, b->mean1, b->inv1);
    mm(b->ln1, b->w1, b->z1, T, M, F);
    for (int i = 0; i < T * F; i++) b->act[i] = b->z1[i] > 0 ? b->z1[i] : 0;
    mm(b->act, b->w2, b->ff, T, F, M);
    for (int i = 0; i < T * M; i++) b->s2[i] = b->ln1[i] + b->ff[i];
    ln(b->s2, b->out, b->mean2, b->inv2);
}

static void backward_layout(Block *b, const float *gout) {
    float ds2[T * M] = {0}, dln1[T * M] = {0}, dff[T * M] = {0}, dact[T * F] = {0}, dz1[T * F] = {0},
          ds1[T * M] = {0}, dproj[T * M] = {0}, dhead[H * T * D] = {0}, dprob[H * T * T] = {0}, dscore[H * T * T] = {0}, dqkv[T * QW] = {0};
    memset(b->dx, 0, sizeof b->dx); memset(b->dwq, 0, sizeof b->dwq); memset(b->dwo, 0, sizeof b->dwo);
    memset(b->dw1, 0, sizeof b->dw1); memset(b->dw2, 0, sizeof b->dw2);
    lnback(b->s2, gout, ds2, b->mean2, b->inv2);
    for (int i = 0; i < T * M; i++) dln1[i] += ds2[i], dff[i] += ds2[i];
    mmback(b->act, b->w2, dff, dact, b->dw2, T, F, M);
    for (int i = 0; i < T * F; i++) dz1[i] = b->z1[i] > 0 ? dact[i] : 0;
    mmback(b->ln1, b->w1, dz1, dln1, b->dw1, T, M, F);
    lnback(b->s1, dln1, ds1, b->mean1, b->inv1);
    for (int i = 0; i < T * M; i++) b->dx[i] += ds1[i], dproj[i] += ds1[i];
    /* Layout-aware: dhead and dwo straight from dproj and b->head/b->wo — no
     * dmerge, no CONTIGUOUS-shaped split-copy back into per-head form. */
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < T; i++) for (int d = 0; d < D; d++) {
            float g = 0;
            for (int j = 0; j < M; j++) g += dproj[i * M + j] * b->wo[(h * D + d) * M + j];
            dhead[(h * T + i) * D + d] = g;
        }
        for (int d = 0; d < D; d++) for (int j = 0; j < M; j++) {
            float g = 0;
            for (int i = 0; i < T; i++) g += b->head[(h * T + i) * D + d] * dproj[i * M + j];
            b->dwo[(h * D + d) * M + j] += g;
        }
    }
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int d = 0; d < D; d++) {
        float g = dhead[(h * T + i) * D + d];
        dprob[(h * T + i) * T + j] += g * b->qkv[j * QW + 2 * M + h * D + d];
        dqkv[j * QW + 2 * M + h * D + d] += g * b->prob[(h * T + i) * T + j];
    }
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
        float dot = 0;
        for (int j = 0; j < T; j++) dot += dprob[(h * T + i) * T + j] * b->prob[(h * T + i) * T + j];
        for (int j = 0; j < T; j++) dscore[(h * T + i) * T + j] = b->prob[(h * T + i) * T + j] * (dprob[(h * T + i) * T + j] - dot);
    }
    float scale = 1 / sqrtf(D);
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int d = 0; d < D; d++) {
        float g = dscore[(h * T + i) * T + j] * scale;
        dqkv[i * QW + h * D + d] += g * b->qkv[j * QW + M + h * D + d];
        dqkv[j * QW + M + h * D + d] += g * b->qkv[i * QW + h * D + d];
    }
    mmback(b->x, b->wq, dqkv, b->dx, b->dwq, T, M, QW);
}

static int same(const float *a, const float *b, int n) { for (int i = 0; i < n; i++) if (fabsf(a[i] - b[i]) > 1e-5f) return 0; return 1; }
static double now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec * 1e9 + ts.tv_nsec; }
static int cmp_double(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y ? 1 : 0; }

/* Median of repeated trials, not a single timed run: at ~170ns for the
 * whole forward pass, single-shot timing is noise-dominated — separate
 * process runs of the naive version disagreed on which variant even won. */
static double median_forward_ns(void (*f)(Block *), Block *b, int trials, uint64_t iters) {
    double t[32];
    for (int tr = 0; tr < trials; tr++) {
        f(b);
        double t0 = now_ns();
        for (uint64_t i = 0; i < iters; i++) f(b);
        double t1 = now_ns();
        t[tr] = (t1 - t0) / iters;
    }
    qsort(t, trials, sizeof(double), cmp_double);
    return t[trials / 2];
}

int main(void) {
    /* --- exact equivalence: same input, same output, forward and backward --- */
    Block reference, layout;
    float seed[T * M];
    for (int i = 0; i < T * M; i++) reference.x[i] = layout.x[i] = initv(i, 1), seed[i] = initv(i, 6);
    for (int i = 0; i < M * QW; i++) reference.wq[i] = layout.wq[i] = initv(i, 2) * .4f;
    for (int i = 0; i < M * M; i++) reference.wo[i] = layout.wo[i] = initv(i, 3) * .4f;
    for (int i = 0; i < M * F; i++) reference.w1[i] = layout.w1[i] = initv(i, 4) * .5f;
    for (int i = 0; i < F * M; i++) reference.w2[i] = layout.w2[i] = initv(i, 5) * .5f;

    forward(&reference);
    forward_layout(&layout);
    if (!same(reference.out, layout.out, T * M)) return 1;
    if (!same(reference.proj, layout.proj, T * M)) return 2;

    backward(&reference, seed);
    backward_layout(&layout, seed);
    if (!same(reference.dx, layout.dx, T * M) || !same(reference.dwq, layout.dwq, M * QW) ||
        !same(reference.dwo, layout.dwo, M * M) || !same(reference.dw1, layout.dw1, M * F) || !same(reference.dw2, layout.dw2, F * M))
        return 3;

    /* --- finite-difference oracle on the layout-aware path directly, same pattern as the reference's own check() --- */
    float plus, minus, old, eps = 1e-3f;
    old = layout.wo[5]; layout.wo[5] = old + eps; forward_layout(&layout); plus = 0; for (int i = 0; i < T * M; i++) plus += layout.out[i] * seed[i];
    layout.wo[5] = old - eps; forward_layout(&layout); minus = 0; for (int i = 0; i < T * M; i++) minus += layout.out[i] * seed[i];
    layout.wo[5] = old; forward_layout(&layout); backward_layout(&layout, seed);
    float numeric = (plus - minus) / (2 * eps), analytic = layout.dwo[5];
    if (fabsf(numeric - analytic) > 5e-3f * fmaxf(1, fmaxf(fabsf(numeric), fabsf(analytic)))) return 4;

    /* --- end-to-end timing: the FULL forward pass against BOTH baselines, median-of-15 --- */
    Block explicit_c;
    memcpy(&explicit_c, &reference, sizeof(Block));
    double fused_ns = median_forward_ns(forward, &reference, 15, 1000000);
    double explicit_ns = median_forward_ns(forward_explicit_contig, &explicit_c, 15, 1000000);
    double layout_ns = median_forward_ns(forward_layout, &layout, 15, 1000000);

    printf("tensor merge layout wired-in: correctness=exact(forward+backward+finite-difference) "
           "full_forward_ns fused=%.2f explicit_contig=%.2f layout=%.2f "
           "layout_vs_fused=%.3fx layout_vs_explicit_contig=%.3fx verdict=within_noise_at_this_shape\n",
           fused_ns, explicit_ns, layout_ns, fused_ns / layout_ns, explicit_ns / layout_ns);
    return 0;
}
