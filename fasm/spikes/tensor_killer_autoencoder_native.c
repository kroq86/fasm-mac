/* Native arm64 deployment runner for the killer-workload's fourth model
 * type: an MLP autoencoder (784->32->784) trained to reconstruct only
 * digit '0' from real MNIST — a genuinely different TASK, not just a
 * different dataset: unsupervised reconstruction, evaluated as anomaly
 * detection (digit '0' = normal, any other digit = anomaly), not
 * classification. Same graph shape as the MNIST MLP killer runner
 * (MATMUL/BIAS_ADD/RELU/MSE, unchanged) with OUT=784 instead of 10.
 *
 * "accuracy" here means anomaly-detection accuracy, computed with NO
 * external threshold artifact (which would have broken the shared 6-file
 * CLI every other killer runner shares): two passes over the fixed
 * TEST_N-sample test set (half real '0's, half random non-'0' digits) —
 * first pass computes each sample's reconstruction MSE and takes the
 * *median* over the whole set as a self-consistent decision boundary
 * (no external state, so it can't drift out of sync between engines);
 * second pass classifies against that.  Same CLI shape and output format
 * as tensor_killer_mnist_native.c so tensor_killer_compare.py /
 * tensor_killer_lifetime_matrix.py work unmodified. */
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include "tensor_semantic_compiler.h"

enum { IN = 784, HID = 64, OUT = 784, TEST_N = 200 };
enum { NX, NW1, NB1, NW2, NB2, NMM1, NADD1, NACT, NMM2, NADD2, NNODES };
int tensor_transformer_steps_execute(const ExecStep *s, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) { int rc = s[i].run(s[i].context); if (rc) return rc; }
    return 0;
}
typedef struct { float x[IN], mm1[HID], add1[HID], act[HID], mm2[OUT], out[OUT]; float gx[IN], gw1[IN * HID], gb1[HID], gw2[HID * OUT], gb2[OUT], gmm1[HID], gadd1[HID], gact[HID], gmm2[OUT], gout[OUT]; Node g[NNODES]; ExecStep steps[32]; Context ctx[32]; uint32_t count; } Runtime;
static int load(const char *p, float *x, size_t n) { FILE *f = fopen(p, "rb"); if (!f) return -1; int ok = fread(x, sizeof(float), n, f) == n && fgetc(f) == EOF; fclose(f); return ok ? 0 : -1; }
static int runtime_init(Runtime *r, float *w1, float *b1, float *w2, float *b2) {
    Node g[NNODES] = {{LEAF, NONE, NONE, INPUT, {r->x, r->gx, NULL, 1, IN, 0}}, {LEAF, NONE, NONE, PARAM, {w1, r->gw1, NULL, IN, HID, 0}}, {LEAF, NONE, NONE, PARAM, {b1, r->gb1, NULL, 1, HID, 0}}, {LEAF, NONE, NONE, PARAM, {w2, r->gw2, NULL, HID, OUT, 0}}, {LEAF, NONE, NONE, PARAM, {b2, r->gb2, NULL, 1, OUT, 0}}, {MATMUL, NX, NW1, TEMP, {r->mm1, r->gmm1, NULL, 1, HID, 0}}, {BIAS_ADD, NMM1, NB1, TEMP, {r->add1, r->gadd1, NULL, 1, HID, 0}}, {RELU, NADD1, NONE, TEMP, {r->act, r->gact, NULL, 1, HID, 0}}, {MATMUL, NACT, NW2, TEMP, {r->mm2, r->gmm2, NULL, 1, OUT, 0}}, {BIAS_ADD, NMM2, NB2, TEMP, {r->out, r->gout, NULL, 1, OUT, 0}}};
    memcpy(r->g, g, sizeof g);
    return compile(r->g, NNODES, 0, r->steps, r->ctx, 32, &r->count) || r->count != 23;
}
static void infer(Runtime *r, const float *x, float *out) { memcpy(r->x, x, IN * sizeof(float)); if (tensor_transformer_steps_execute(r->steps, 5)) abort(); memcpy(out, r->out, OUT * sizeof(float)); }
static double ns(uint64_t d) { mach_timebase_info_data_t t; mach_timebase_info(&t); return (double)d * t.numer / t.denom; }
static int cmp64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    if (argc != 8) { fprintf(stderr, "usage: %s w1 b1 w2 b2 inputs labels reps\n", argv[0]); return 2; }
    unsigned reps = (unsigned)strtoul(argv[7], 0, 10);
    int skip = getenv("KILLER_SKIP_ACCURACY") != NULL;
    float *w1 = malloc(IN * HID * 4), *b1 = malloc(HID * 4), *w2 = malloc(HID * OUT * 4), *b2 = malloc(OUT * 4), *x = malloc((size_t)TEST_N * IN * 4);
    uint64_t *dt = malloc((size_t)reps * sizeof *dt);
    unsigned char *y = malloc(TEST_N);
    Runtime *r = calloc(1, sizeof *r);
    if (!w1 || !b1 || !w2 || !b2 || !x || !dt || !y || !r || load(argv[1], w1, IN * HID) || load(argv[2], b1, HID) || load(argv[3], w2, HID * OUT) || load(argv[4], b2, OUT) || load(argv[5], x, (size_t)TEST_N * IN) || runtime_init(r, w1, b1, w2, b2)) {
        fprintf(stderr, "model/runtime load failed\n");
        return 3;
    }
    FILE *f = fopen(argv[6], "rb");
    if (!f || fread(y, 1, TEST_N, f) != TEST_N) { fprintf(stderr, "label load failed\n"); return 3; }
    fclose(f);
    float out[OUT], sum = 0;
    unsigned checks = skip ? 1 : TEST_N;
    unsigned correct = 0;
    if (!skip) {
        /* mean, not median: the median of a sorted array IS one of the
         * test set's own MSE values, which guarantees at least one sample
         * sits exactly ON the decision boundary — maximally sensitive to
         * float-summation-order differences between engines (confirmed:
         * an earlier median-based version gave 0.375 here vs numpy's
         * 0.370, a single borderline sample flipping). The mean is not
         * generally equal to any one sample's value, so a knife-edge tie
         * is far less likely in practice. */
        float *mse = malloc(TEST_N * sizeof(float));
        double mse_sum = 0;
        for (unsigned n = 0; n < TEST_N; n++) {
            infer(r, x + (size_t)n * IN, out);
            float e = 0;
            for (int k = 0; k < IN; k++) { float d = out[k] - x[(size_t)n * IN + k]; e += d * d; sum += out[k]; }
            mse[n] = e / IN;
            mse_sum += mse[n];
        }
        float threshold = (float)(mse_sum / TEST_N);
        for (unsigned n = 0; n < TEST_N; n++) correct += (mse[n] > threshold) == (y[n] != 0);
        free(mse);
    } else {
        infer(r, x, out);
        for (int k = 0; k < OUT; k++) sum += out[k];
    }
    for (unsigned q = 0; q < reps; q++) {
        uint64_t t = mach_absolute_time();
        infer(r, x + (size_t)(q % TEST_N) * IN, out);
        dt[q] = mach_absolute_time() - t;
    }
    qsort(dt, reps, sizeof *dt, cmp64);
    volatile float sink = sum;
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    printf("engine\tnative\nwarm_median_ns\t%.3f\naccuracy\t%.6f\npeak_rss_bytes\t%ld\nchecksum\t%.9g\n",
           ns(dt[reps / 2]), correct / (double)checks, (long)u.ru_maxrss, (double)sink);
    return 0;
}
