/* Experimental size-aware SIMD profiling spike; not a stable core ABI.
 *
 * The block-level executor benchmark (tensor_transformer_executor_kernels_check.c)
 * showed NEON losing to scalar on one fixed small shape (T=3 tokens). This
 * measures scalar vs native (NEON on arm64, AVX2 on x86_64) across a real
 * size sweep, for every family that actually has two implementations to
 * switch between: matmul, residual, bias, relu, zero, sgd. Attention,
 * softmax, LayerNorm, and permutation have no SIMD alternative anywhere in
 * this repo yet, so no crossover can be measured for them here.
 *
 * Compiled two ways, matching tensor_neon_check.c's convention: plain -O3
 * (scalar_* may get auto-vectorized by clang's own optimizer, so this is
 * really "hand NEON vs whatever the compiler gives you for free") and
 * -fno-vectorize -fno-slp-vectorize (scalar_* is truly scalar). The two
 * modes can disagree completely, and the difference is the finding: it
 * separates "does SIMD help at all" from "does OUR hand-written NEON beat
 * the compiler's own auto-vectorization of equivalent C".
 */
#ifndef SCALAR_MODE
#define SCALAR_MODE "compiler_optimized"
#endif
#define TENSOR_DISPATCH_NO_MAIN
#include "tensor_kernel_dispatch_spike.h"
#include <time.h>

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static uint64_t clampu(uint64_t v, uint64_t lo, uint64_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static double time_call(void (*call)(void *), void *ctx, uint64_t iters) {
    call(ctx); /* warm up: touch pages, prime branch predictor/caches */
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) call(ctx);
    double t1 = now_ns();
    return (t1 - t0) / (double)iters;
}

typedef struct { Unary f; const float *a; float *o; uint64_t n; } UnaryCtx;
static void call_unary(void *p) { UnaryCtx *c = p; c->f(c->a, c->o, c->n); }
typedef struct { Binary f; const float *a, *b; float *o; uint64_t n; } BinaryCtx;
static void call_binary(void *p) { BinaryCtx *c = p; c->f(c->a, c->b, c->o, c->n); }
typedef struct { Bias f; const float *a, *b; float *o; uint64_t r, c; } BiasCtx;
static void call_bias(void *p) { BiasCtx *x = p; x->f(x->a, x->b, x->o, x->r, x->c); }
typedef struct { Sgd f; float *p; const float *g; uint64_t n; } SgdCtx;
static void call_sgd(void *p) { SgdCtx *c = p; c->f(c->p, c->g, 0.0f, c->n); } /* lr=0: measure cost without numerical drift */
typedef struct { Matmul f; const float *a, *b; float *o; uint64_t m, k, n; } MatmulCtx;
static void call_matmul(void *p) { MatmulCtx *c = p; c->f(c->a, c->b, c->o, c->m, c->k, c->n); }

static void report(const char *family, uint64_t size, double scalar_ns, double native_ns, const char *native_name) {
    double speedup = scalar_ns / native_ns;
    printf("family=%-8s size=%6llu scalar_ns=%9.1f %s_ns=%9.1f speedup=%.2fx %s\n",
           family, (unsigned long long)size, scalar_ns, native_name, native_ns, speedup,
           speedup > 1.0 ? "native_wins" : "scalar_wins");
}

int main(void) {
    Dispatch s = scalar_dispatch();
    if (!native_available()) {
        printf("tensor kernel dispatch profile: no native backend available on this build (backend=%s), nothing to compare\n", NATIVE_NAME);
        return 0;
    }
    Dispatch d = auto_dispatch();

    uint64_t elementwise_sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    unsigned n_elementwise = sizeof elementwise_sizes / sizeof elementwise_sizes[0];
    uint64_t max_n = elementwise_sizes[n_elementwise - 1];

    float *a = malloc(max_n * 4), *b = malloc(max_n * 4), *o = malloc(max_n * 4);
    for (uint64_t i = 0; i < max_n; i++) { a[i] = v(i, 1); b[i] = v(i, 2); }

    int64_t first_native_win_residual = -1, first_native_win_relu = -1, first_native_win_zero = -1, first_native_win_bias = -1, first_native_win_sgd = -1;

    for (unsigned z = 0; z < n_elementwise; z++) {
        uint64_t n = elementwise_sizes[z];
        uint64_t iters = clampu(50000000ull / n, 50, 500000);

        BinaryCtx rc_s = {s.residual, a, b, o, n}, rc_d = {d.residual, a, b, o, n};
        double rs = time_call(call_binary, &rc_s, iters), rn = time_call(call_binary, &rc_d, iters);
        report("residual", n, rs, rn, d.backend);
        if (first_native_win_residual < 0 && rn < rs) first_native_win_residual = (int64_t)n;

        UnaryCtx lu_s = {s.relu, a, o, n}, lu_d = {d.relu, a, o, n};
        double ls = time_call(call_unary, &lu_s, iters), ln = time_call(call_unary, &lu_d, iters);
        report("relu", n, ls, ln, d.backend);
        if (first_native_win_relu < 0 && ln < ls) first_native_win_relu = (int64_t)n;

        UnaryCtx zu_s = {s.zero, a, o, n}, zu_d = {d.zero, a, o, n};
        double zs = time_call(call_unary, &zu_s, iters), zn = time_call(call_unary, &zu_d, iters);
        report("zero", n, zs, zn, d.backend);
        if (first_native_win_zero < 0 && zn < zs) first_native_win_zero = (int64_t)n;

        uint64_t rows = 8, cols = n / rows ? n / rows : 1;
        uint64_t biters = clampu(50000000ull / (rows * cols), 50, 500000);
        BiasCtx bc_s = {s.bias, a, b, o, rows, cols}, bc_d = {d.bias, a, b, o, rows, cols};
        double bs = time_call(call_bias, &bc_s, biters), bn = time_call(call_bias, &bc_d, biters);
        report("bias", n, bs, bn, d.backend);
        if (first_native_win_bias < 0 && bn < bs) first_native_win_bias = (int64_t)n;

        SgdCtx sc_s = {s.sgd, o, b, n}, sc_d = {d.sgd, o, b, n};
        double ss = time_call(call_sgd, &sc_s, iters), sn = time_call(call_sgd, &sc_d, iters);
        report("sgd", n, ss, sn, d.backend);
        if (first_native_win_sgd < 0 && sn < ss) first_native_win_sgd = (int64_t)n;
    }

    /* matmul: sweep both the column count (n=k, square) and the row count m.
     * m=3 is the real transformer block's token count (T=3) — the exact
     * shape the block-level benchmark measured — alongside larger m to show
     * whether row count, not just column count, changes the outcome. */
    uint64_t matmul_sizes[] = {4, 8, 12, 16, 32, 64, 128, 256, 512, 1024};
    unsigned n_matmul = sizeof matmul_sizes / sizeof matmul_sizes[0];
    uint64_t matmul_rows[] = {3, 8, 64};
    unsigned n_rows = sizeof matmul_rows / sizeof matmul_rows[0];
    uint64_t max_mm = matmul_sizes[n_matmul - 1], max_m = matmul_rows[n_rows - 1];
    float *ma = malloc(max_m * max_mm * 4), *mb = malloc(max_mm * max_mm * 4), *mo = malloc(max_m * max_mm * 4);
    for (uint64_t i = 0; i < max_m * max_mm; i++) ma[i] = v(i, 3);
    for (uint64_t i = 0; i < max_mm * max_mm; i++) mb[i] = v(i, 4);
    int64_t first_native_win_matmul_m3 = -1;
    for (unsigned r = 0; r < n_rows; r++) {
        uint64_t m = matmul_rows[r];
        for (unsigned z = 0; z < n_matmul; z++) {
            uint64_t n = matmul_sizes[z];
            uint64_t work = m * n * n;
            uint64_t iters = clampu(2000000000ull / work, 1, 20000);
            MatmulCtx mc_s = {s.matmul, ma, mb, mo, m, n, n}, mc_d = {d.matmul, ma, mb, mo, m, n, n};
            double ms = time_call(call_matmul, &mc_s, iters), mn = time_call(call_matmul, &mc_d, iters);
            char label[16];
            snprintf(label, sizeof label, "matmul_m%llu", (unsigned long long)m);
            report(label, n, ms, mn, d.backend);
            if (m == 3 && first_native_win_matmul_m3 < 0 && mn < ms) first_native_win_matmul_m3 = (int64_t)n;
        }
    }

    printf("scalar_mode=%s\n", SCALAR_MODE);
    printf("crossover(size k where %s first beats scalar; -1 = never in tested range; matmul at m=3, the real block token count):\n", d.backend);
    printf("  matmul_m3=%lld residual=%lld relu=%lld zero=%lld bias=%lld sgd=%lld\n",
           (long long)first_native_win_matmul_m3, (long long)first_native_win_residual, (long long)first_native_win_relu,
           (long long)first_native_win_zero, (long long)first_native_win_bias, (long long)first_native_win_sgd);
    printf("no SIMD alternative exists yet for: attention, softmax, layernorm, permutation/contiguous — not profiled here\n");
    return 0;
}
