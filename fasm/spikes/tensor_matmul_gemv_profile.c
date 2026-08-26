/* Specialized M=1 GEMV kernel: correctness + isolated benchmark against
 * production -O3 scalar C and against the ikj/vectorize-over-N NEON kernel
 * that tensor_matmul_scaling_profile.c already showed losing at M=1; not a
 * stable core ABI.
 *
 * Design, chosen specifically to fix what lost last time: the ikj kernel
 * re-read/re-wrote the *entire* output row from memory on every one of the
 * K reduction steps, because it never kept an accumulator resident in a
 * register across K. This kernel blocks the output into 4-wide columns
 * and, for each block, runs the *whole* K reduction with the accumulator
 * held in a NEON register, writing the output exactly once at the end of
 * that block's K loop. Net effect: B is still read exactly once overall
 * (each element touched by exactly one (K,block) pair), the output is
 * written exactly once overall (not K times), and A (K=784 floats, ~3KB)
 * is small enough to stay L1-resident across every block's re-read of it —
 * this is the standard blocked-GEMV shape, not a novel idea, but it hadn't
 * been measured against this repo's actual M=1 inference shapes before.
 *
 * Numerical equivalence: verified against tensor_kernel_dispatch_spike.h's
 * scalar_matmul with the same eq() tolerance (3e-5 relative) that file's
 * own check() already uses, at every shape benchmarked below, before any
 * timing happens.
 */
#define TENSOR_DISPATCH_NO_MAIN
#include "tensor_kernel_dispatch_spike.h"
#include <mach/mach_time.h>

static void gemv_matmul(const float *a, const float *b, float *o, uint64_t m, uint64_t k, uint64_t n) {
    for (uint64_t i = 0; i < m; i++) {
        const float *arow = a + i * k;
        float *orow = o + i * n;
        uint64_t j = 0;
#if defined(__aarch64__)
        for (; j + 4 <= n; j += 4) {
            float32x4_t acc = vdupq_n_f32(0);
            for (uint64_t p = 0; p < k; p++) acc = vfmaq_f32(acc, vdupq_n_f32(arow[p]), vld1q_f32(b + p * n + j));
            vst1q_f32(orow + j, acc);
        }
#endif
        for (; j < n; j++) {
            float s = 0;
            for (uint64_t p = 0; p < k; p++) s += arow[p] * b[p * n + j];
            orow[j] = s;
        }
    }
}

static double ns(void) {
    static mach_timebase_info_data_t t;
    if (!t.denom) mach_timebase_info(&t);
    return (double)mach_absolute_time() * t.numer / t.denom;
}
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
typedef struct { double median, min, max; } Stats;
static Stats time_matmul(Matmul f, const float *a, const float *b, float *o, uint64_t m, uint64_t k, uint64_t n, unsigned reps) {
    double *t = malloc(reps * sizeof *t);
    f(a, b, o, m, k, n); /* warm up */
    for (unsigned r = 0; r < reps; r++) {
        double t0 = ns();
        f(a, b, o, m, k, n);
        t[r] = ns() - t0;
    }
    qsort(t, reps, sizeof *t, cmpd);
    Stats s = {t[reps / 2], t[0], t[reps - 1]};
    free(t);
    return s;
}

int main(void) {
    if (!native_available()) {
        printf("matmul gemv profile: no native backend available on this build (backend=%s), nothing to compare\n", NATIVE_NAME);
        return 0;
    }
    uint64_t IN = 784, OUT = 10;
    uint64_t hids[] = {32, 126, 630, 1258, 2516, 5030};
    unsigned n_hid = sizeof hids / sizeof hids[0];
    uint64_t max_k = IN > hids[n_hid - 1] ? IN : hids[n_hid - 1];
    uint64_t max_n = hids[n_hid - 1];
    float *a = malloc(max_k * 4), *b = malloc(max_k * max_n * 4);
    float *o_ref = malloc(max_n * 4), *o_neon = malloc(max_n * 4), *o_gemv = malloc(max_n * 4);
    for (uint64_t i = 0; i < max_k; i++) a[i] = v(i, 1);
    for (uint64_t i = 0; i < max_k * max_n; i++) b[i] = v(i, 2);

    int mismatches = 0;
    unsigned reps = 200;
    unsigned wins = 0, losses = 0;
    printf("engine=neon(ikj) vs gemv(blocked) vs scalar, all M=1, reps=%u, times in ns (median [min,max])\n", reps);
    for (unsigned z = 0; z < n_hid; z++) {
        uint64_t hid = hids[z];
        struct { uint64_t k, n; const char *label; } shapes[] = {{IN, hid, "layer1"}, {hid, OUT, "layer2"}};
        for (unsigned s = 0; s < 2; s++) {
            uint64_t k = shapes[s].k, n = shapes[s].n;
            scalar_matmul(a, b, o_ref, 1, k, n);
            native_matmul(a, b, o_neon, 1, k, n);
            gemv_matmul(a, b, o_gemv, 1, k, n);
            if (!eq(o_ref, o_neon, n)) { fprintf(stderr, "gemv profile: %s hid=%llu neon(ikj) mismatch\n", shapes[s].label, (unsigned long long)hid); mismatches++; }
            if (!eq(o_ref, o_gemv, n)) { fprintf(stderr, "gemv profile: %s hid=%llu gemv mismatch\n", shapes[s].label, (unsigned long long)hid); mismatches++; }

            Stats ts = time_matmul(scalar_matmul, a, b, o_ref, 1, k, n, reps);
            Stats tn = time_matmul(native_matmul, a, b, o_neon, 1, k, n, reps);
            Stats tg = time_matmul(gemv_matmul, a, b, o_gemv, 1, k, n, reps);
            printf("hid=%-5llu %-6s shape=1x%llux%-5llu scalar=%9.1f[%7.1f,%9.1f] neon_ikj=%9.1f[%7.1f,%9.1f] "
                   "gemv=%9.1f[%7.1f,%9.1f] gemv_vs_scalar=%.2fx %s\n",
                   (unsigned long long)hid, shapes[s].label, (unsigned long long)k, (unsigned long long)n,
                   ts.median, ts.min, ts.max, tn.median, tn.min, tn.max, tg.median, tg.min, tg.max,
                   ts.median / tg.median, tg.median < ts.median ? "gemv_wins" : "scalar_wins");
            if (shapes[s].label[0] == 'l' && shapes[s].label[5] == '1') { /* layer1: the dominant, decisive shape */
                if (tg.median < ts.median) wins++; else losses++;
            }
        }
    }
    printf("dominant-shape (layer1) verdict: gemv_wins=%u scalar_wins=%u out_of=%u -> %s\n",
           wins, losses, wins + losses, wins == n_hid ? "SPECIALIZE" : "DO NOT SPECIALIZE");
    if (mismatches) { fprintf(stderr, "gemv profile: %d numerical mismatch(es), refusing to recommend specialization\n", mismatches); return 1; }
    return 0;
}
