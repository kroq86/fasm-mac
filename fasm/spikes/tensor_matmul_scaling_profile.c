/* MATMUL scalar-vs-NEON microbenchmark at the exact (M=1,K,N) shapes the
 * killer-workload scaling matrix (tensor_killer_scaling_native.c, ~25k to
 * ~4M param MLP family) actually uses — not a generic square sweep. Both
 * layers of that MLP are batch=1 (M=1): layer 1 is (1,784,HID), layer 2 is
 * (1,HID,10), for HID in {32,126,630,1258,2516,5030}.
 *
 * Compiled plain -O3 (not -fno-vectorize) deliberately: the question this
 * answers is "does hand NEON beat what the compiler already does for free
 * in the actual build the killer benchmark uses", matching
 * tensor_kernel_dispatch_profile.c's "compiler_optimized" convention,
 * since -O3 plain is what check_tensor_killer_scaling_matrix.sh actually
 * compiles with.
 *
 * No new kernel code: reuses tensor_kernel_dispatch_spike.h's
 * already-verified scalar_matmul/native_matmul unchanged, and additionally
 * checks correctness at each of these specific real shapes (the header's
 * own check() only covers square tail shapes 1..17) before timing them. */
#define TENSOR_DISPATCH_NO_MAIN
#include "tensor_kernel_dispatch_spike.h"
#include <time.h>

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
static double time_matmul(Matmul f, const float *a, const float *b, float *o, uint64_t m, uint64_t k, uint64_t n, uint64_t iters) {
    f(a, b, o, m, k, n);
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) f(a, b, o, m, k, n);
    double t1 = now_ns();
    return (t1 - t0) / (double)iters;
}

int main(void) {
    if (!native_available()) {
        printf("matmul scaling profile: no native backend available on this build (backend=%s), nothing to compare\n", NATIVE_NAME);
        return 0;
    }
    uint64_t IN = 784, OUT = 10;
    uint64_t hids[] = {32, 126, 630, 1258, 2516, 5030};
    unsigned n_hid = sizeof hids / sizeof hids[0];
    uint64_t max_k = IN > hids[n_hid - 1] ? IN : hids[n_hid - 1];
    uint64_t max_n = hids[n_hid - 1];
    float *a = malloc(max_k * 4), *b = malloc(max_k * max_n * 4), *o_s = malloc(max_n * 4), *o_n = malloc(max_n * 4);
    for (uint64_t i = 0; i < max_k; i++) a[i] = v(i, 1);
    for (uint64_t i = 0; i < max_k * max_n; i++) b[i] = v(i, 2);

    int mismatches = 0;
    for (unsigned z = 0; z < n_hid; z++) {
        uint64_t hid = hids[z];
        struct { uint64_t k, n; const char *label; } shapes[] = {{IN, hid, "layer1"}, {hid, OUT, "layer2"}};
        for (unsigned s = 0; s < 2; s++) {
            uint64_t k = shapes[s].k, n = shapes[s].n;
            scalar_matmul(a, b, o_s, 1, k, n);
            native_matmul(a, b, o_n, 1, k, n);
            if (!eq(o_s, o_n, n)) { fprintf(stderr, "matmul scaling profile: %s hid=%llu correctness mismatch\n", shapes[s].label, (unsigned long long)hid); mismatches++; }
            uint64_t work = k * n;
            uint64_t iters = work < 2000 ? 2000000 : (work < 200000 ? 200000 : 20000);
            double ts = time_matmul(scalar_matmul, a, b, o_s, 1, k, n, iters);
            double tn = time_matmul(native_matmul, a, b, o_n, 1, k, n, iters);
            printf("hid=%-5llu %-6s shape=1x%llux%llu scalar_ns=%9.1f %s_ns=%9.1f speedup=%.2fx %s\n",
                   (unsigned long long)hid, shapes[s].label, (unsigned long long)k, (unsigned long long)n,
                   ts, NATIVE_NAME, tn, ts / tn, ts > tn ? "native_wins" : "scalar_wins");
        }
    }
    printf("scalar_mode=compiler_optimized(-O3, not -fno-vectorize) backend=%s\n", NATIVE_NAME);
    return mismatches ? 1 : 0;
}
