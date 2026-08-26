/* Native arm64 deployment runner for the killer-workload's third model
 * type: a small tabular classifier (13->16->3 MLP) on the real UCI Wine
 * dataset (178 samples, 13 physicochemical features, 3 cultivar classes —
 * https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data),
 * batch=1. Same graph topology as the MNIST MLP killer runner — this file
 * exists mainly because the test-set size differs (36, not 1000) and the
 * shapes are compile-time constants matching this specific model, not
 * because tabular data needs different ops: MATMUL/BIAS_ADD/RELU/MSE,
 * unchanged, same as every other killer runner. Same CLI shape and output
 * format as tensor_killer_mnist_native.c (w1 b1 w2 b2 inputs labels reps)
 * so tensor_killer_compare.py / tensor_killer_lifetime_matrix.py work
 * unmodified. */
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include "tensor_semantic_compiler.h"

enum { IN = 13, HID = 16, OUT = 3, TEST_N = 36 };
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
    float *w1 = malloc(IN * HID * 4), *b1 = malloc(HID * 4), *w2 = malloc(HID * OUT * 4), *b2 = malloc(OUT * 4), *x = malloc(TEST_N * IN * 4);
    uint64_t *dt = malloc((size_t)reps * sizeof *dt);
    unsigned char *y = malloc(TEST_N);
    Runtime *r = calloc(1, sizeof *r);
    if (!w1 || !b1 || !w2 || !b2 || !x || !dt || !y || !r || load(argv[1], w1, IN * HID) || load(argv[2], b1, HID) || load(argv[3], w2, HID * OUT) || load(argv[4], b2, OUT) || load(argv[5], x, TEST_N * IN) || runtime_init(r, w1, b1, w2, b2)) {
        fprintf(stderr, "model/runtime load failed\n");
        return 3;
    }
    FILE *f = fopen(argv[6], "rb");
    if (!f || fread(y, 1, TEST_N, f) != TEST_N) { fprintf(stderr, "label load failed\n"); return 3; }
    fclose(f);
    float out[OUT], sum = 0;
    unsigned correct = 0, checks = skip ? 1 : TEST_N;
    for (unsigned n = 0; n < checks; n++) {
        infer(r, x + (size_t)n * IN, out);
        int p = 0;
        for (int k = 0; k < OUT; k++) { sum += out[k]; if (out[k] > out[p]) p = k; }
        correct += p == y[n];
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
    return !skip && correct < (checks * 3) / 4 ? 1 : 0;
}
