/* Experimental layout-aware kernel spike; not a stable core ABI.
 *
 * tensor_qkv_layout_check.c already established that merging attention
 * heads [H,T,D] -> [T,H*D] is not a legal zero-copy reshape, so the planner
 * always inserts an explicit CONTIGUOUS action before the output
 * projection matmul. This asks the narrower, concrete question: does a
 * kernel that reads the [H,T,D] layout directly (skipping the copy
 * entirely) actually win, instead of assuming it and building a general
 * layout-aware dispatcher first.
 *
 * The reformulation is exact, not approximate: head is stored [H][T][D]
 * (h outermost). For fixed h, the [T,D] slice head+h*T*D is already an
 * ordinary contiguous block, and so is wo's [D,M] row-slice wo+h*D*M (wo is
 * row-major [M,M], M=H*D). So proj[T,M] = sum_h head_h[T,D] @ wo_h[D,M] is
 * mathematically identical to (contiguous-copy into merge[T,M]) + one
 * [T,M]x[M,M] matmul — just H accumulating matmuls over natural contiguous
 * sub-blocks instead of one matmul over a materialized copy. No strided
 * "gather" kernel is even needed for this specific case.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static float v(uint64_t i, uint64_t s) { return (float)((int)((i * 41 + s * 13) % 37) - 18) / 23.0f; }
static int closef(float a, float b) { return fabsf(a - b) <= 1e-4f * fmaxf(1, fmaxf(fabsf(a), fabsf(b))); }

static void proj_via_contiguous(const float *head, const float *wo, float *proj, float *merge, int T, int H, int D, int M) {
    for (int i = 0; i < T; i++) for (int h = 0; h < H; h++) for (int d = 0; d < D; d++) merge[i * M + h * D + d] = head[(h * T + i) * D + d];
    memset(proj, 0, (size_t)T * M * sizeof(float));
    for (int i = 0; i < T; i++) for (int k = 0; k < M; k++) { float x = merge[i * M + k]; for (int j = 0; j < M; j++) proj[i * M + j] += x * wo[k * M + j]; }
}

static void proj_via_layout(const float *head, const float *wo, float *proj, int T, int H, int D, int M) {
    memset(proj, 0, (size_t)T * M * sizeof(float));
    for (int h = 0; h < H; h++) {
        const float *head_h = head + (size_t)h * T * D; /* contiguous [T,D] */
        const float *wo_h = wo + (size_t)h * D * M;      /* contiguous [D,M] */
        for (int i = 0; i < T; i++) for (int d = 0; d < D; d++) { float x = head_h[i * D + d]; for (int j = 0; j < M; j++) proj[i * M + j] += x * wo_h[d * M + j]; }
    }
}

static int check_shape(int T, int H, int D) {
    int M = H * D;
    float *head = malloc((size_t)H * T * D * sizeof(float)), *wo = malloc((size_t)M * M * sizeof(float));
    float *proj_a = malloc((size_t)T * M * sizeof(float)), *proj_b = malloc((size_t)T * M * sizeof(float)), *merge = malloc((size_t)T * M * sizeof(float));
    for (int i = 0; i < H * T * D; i++) head[i] = v(i, 1);
    for (int i = 0; i < M * M; i++) wo[i] = v(i, 2);
    proj_via_contiguous(head, wo, proj_a, merge, T, H, D, M);
    proj_via_layout(head, wo, proj_b, T, H, D, M);
    int ok = 1;
    for (int i = 0; i < T * M; i++) if (!closef(proj_a[i], proj_b[i])) ok = 0;
    free(head); free(wo); free(proj_a); free(proj_b); free(merge);
    return ok;
}

static double now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec * 1e9 + ts.tv_nsec; }

static void profile_shape(const char *sweep, int T, int H, int D) {
    int M = H * D;
    float *head = malloc((size_t)H * T * D * sizeof(float)), *wo = malloc((size_t)M * M * sizeof(float));
    float *proj = malloc((size_t)T * M * sizeof(float)), *merge = malloc((size_t)T * M * sizeof(float));
    for (int i = 0; i < H * T * D; i++) head[i] = v(i, 1);
    for (int i = 0; i < M * M; i++) wo[i] = v(i, 2);
    uint64_t work = (uint64_t)T * M * M; /* FLOP-proportional, same for both variants */
    uint64_t iters = work ? 200000000ull / work : 1;
    if (iters < 20) iters = 20;
    if (iters > 500000) iters = 500000;
    proj_via_contiguous(head, wo, proj, merge, T, H, D, M);
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) proj_via_contiguous(head, wo, proj, merge, T, H, D, M);
    double t1 = now_ns();
    proj_via_layout(head, wo, proj, T, H, D, M);
    double t2 = now_ns();
    for (uint64_t i = 0; i < iters; i++) proj_via_layout(head, wo, proj, T, H, D, M);
    double t3 = now_ns();
    double contig_ns = (t1 - t0) / iters, layout_ns = (t3 - t2) / iters;
    printf("sweep=%-6s T=%4d H=%2d D=%3d M=%4d contiguous_ns=%9.1f layout_ns=%9.1f speedup=%.2fx %s\n",
           sweep, T, H, D, M, contig_ns, layout_ns, contig_ns / layout_ns, layout_ns < contig_ns ? "layout_wins" : "contiguous_wins");
    free(head); free(wo); free(proj); free(merge);
}

int main(void) {
    int shapes[][3] = {{3,2,2},{3,2,4},{8,2,2},{16,4,8},{64,8,16}};
    for (unsigned i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
        if (!check_shape(shapes[i][0], shapes[i][1], shapes[i][2])) { fprintf(stderr, "correctness failed T=%d H=%d D=%d\n", shapes[i][0], shapes[i][1], shapes[i][2]); return 1; }

    /* Real block shape first, then three independent scaling sweeps. */
    profile_shape("real", 3, 2, 2);
    int t_sweep[] = {3, 8, 16, 32, 64, 128, 256, 512};
    for (unsigned i = 0; i < sizeof t_sweep / sizeof t_sweep[0]; i++) profile_shape("tokens", t_sweep[i], 2, 2);
    int d_sweep[] = {2, 4, 8, 16, 32, 64, 128};
    for (unsigned i = 0; i < sizeof d_sweep / sizeof d_sweep[0]; i++) profile_shape("headdim", 3, 2, d_sweep[i]);
    int h_sweep[] = {2, 4, 8, 16};
    for (unsigned i = 0; i < sizeof h_sweep / sizeof h_sweep[0]; i++) profile_shape("heads", 3, h_sweep[i], 2);
    printf("tensor merge layout profile: correctness=exact shapes_checked=5 sweeps=tokens,headdim,heads\n");
    return 0;
}
