/* Generic native arm64 deployment runner for the killer-workload scaling
 * matrix. Same [IN,HID,OUT] MLP topology and canonical compile() path as
 * tensor_killer_mnist_native.c — parameterized at runtime instead of
 * hardcoded to one 784->32->10 shape, so one binary covers every point in
 * the scaling matrix. No change to the compiler, op set, or executor:
 * MATMUL/BIAS_ADD/RELU are already fully shape-generic (rows/cols are
 * Node fields, not compile-time macros), which is exactly what makes this
 * possible without touching tensor_semantic_compiler.h. Inference-only. */
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include "tensor_semantic_compiler.h"

enum { NX, NW1, NB1, NW2, NB2, NMM1, NADD1, NACT, NMM2, NADD2, NNODES };
int tensor_transformer_steps_execute(const ExecStep *s, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) { int rc = s[i].run(s[i].context); if (rc) return rc; }
    return 0;
}
typedef struct {
    uint32_t in, hid, out;
    float *x, *mm1, *add1, *act, *mm2, *outp;
    float *gx, *gw1, *gb1, *gw2, *gb2, *gmm1, *gadd1, *gact, *gmm2, *gout;
    Node g[NNODES];
    ExecStep steps[32];
    Context ctx[32];
    uint32_t count;
} Runtime;

static int load(const char *p, float *x, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f) return -1;
    int ok = fread(x, sizeof(float), n, f) == n && fgetc(f) == EOF;
    fclose(f);
    return ok ? 0 : -1;
}
static int runtime_init(Runtime *r, float *w1, float *b1, float *w2, float *b2) {
    uint32_t IN = r->in, HID = r->hid, OUT = r->out;
    r->x = calloc(IN, 4); r->mm1 = calloc(HID, 4); r->add1 = calloc(HID, 4); r->act = calloc(HID, 4);
    r->mm2 = calloc(OUT, 4); r->outp = calloc(OUT, 4);
    r->gx = calloc(IN, 4); r->gw1 = calloc((size_t)IN * HID, 4); r->gb1 = calloc(HID, 4);
    r->gw2 = calloc((size_t)HID * OUT, 4); r->gb2 = calloc(OUT, 4);
    r->gmm1 = calloc(HID, 4); r->gadd1 = calloc(HID, 4); r->gact = calloc(HID, 4);
    r->gmm2 = calloc(OUT, 4); r->gout = calloc(OUT, 4);
    if (!r->x || !r->mm1 || !r->add1 || !r->act || !r->mm2 || !r->outp || !r->gx || !r->gw1 || !r->gb1 ||
        !r->gw2 || !r->gb2 || !r->gmm1 || !r->gadd1 || !r->gact || !r->gmm2 || !r->gout)
        return -1;
    Node g[NNODES] = {
        {LEAF, NONE, NONE, INPUT, {r->x, r->gx, NULL, 1, IN, 0}},
        {LEAF, NONE, NONE, PARAM, {w1, r->gw1, NULL, IN, HID, 0}},
        {LEAF, NONE, NONE, PARAM, {b1, r->gb1, NULL, 1, HID, 0}},
        {LEAF, NONE, NONE, PARAM, {w2, r->gw2, NULL, HID, OUT, 0}},
        {LEAF, NONE, NONE, PARAM, {b2, r->gb2, NULL, 1, OUT, 0}},
        {MATMUL, NX, NW1, TEMP, {r->mm1, r->gmm1, NULL, 1, HID, 0}},
        {BIAS_ADD, NMM1, NB1, TEMP, {r->add1, r->gadd1, NULL, 1, HID, 0}},
        {RELU, NADD1, NONE, TEMP, {r->act, r->gact, NULL, 1, HID, 0}},
        {MATMUL, NACT, NW2, TEMP, {r->mm2, r->gmm2, NULL, 1, OUT, 0}},
        {BIAS_ADD, NMM2, NB2, TEMP, {r->outp, r->gout, NULL, 1, OUT, 0}},
    };
    memcpy(r->g, g, sizeof g);
    return compile(r->g, NNODES, 0, r->steps, r->ctx, 32, &r->count) || r->count != 23;
}
static void infer(Runtime *r, const float *x, float *out) {
    memcpy(r->x, x, r->in * sizeof(float));
    if (tensor_transformer_steps_execute(r->steps, 5)) abort();
    memcpy(out, r->outp, r->out * sizeof(float));
}
static double ns(uint64_t d) { mach_timebase_info_data_t t; mach_timebase_info(&t); return (double)d * t.numer / t.denom; }
static int cmp64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    if (argc != 11) { fprintf(stderr, "usage: %s w1 b1 w2 b2 inputs labels reps IN HID OUT\n", argv[0]); return 2; }
    unsigned reps = (unsigned)strtoul(argv[7], 0, 10);
    int skip = getenv("KILLER_SKIP_ACCURACY") != NULL;
    Runtime *r = calloc(1, sizeof *r);
    if (!r) return 3;
    r->in = (uint32_t)strtoul(argv[8], 0, 10);
    r->hid = (uint32_t)strtoul(argv[9], 0, 10);
    r->out = (uint32_t)strtoul(argv[10], 0, 10);
    uint32_t IN = r->in, HID = r->hid, OUT = r->out;
    float *w1 = malloc((size_t)IN * HID * 4), *b1 = malloc((size_t)HID * 4);
    float *w2 = malloc((size_t)HID * OUT * 4), *b2 = malloc((size_t)OUT * 4);
    float *x = malloc((size_t)1000 * IN * 4), *out = malloc((size_t)OUT * 4);
    uint64_t *dt = malloc((size_t)reps * sizeof *dt);
    unsigned char *y = malloc(1000);
    if (!w1 || !b1 || !w2 || !b2 || !x || !out || !dt || !y ||
        load(argv[1], w1, (size_t)IN * HID) || load(argv[2], b1, HID) ||
        load(argv[3], w2, (size_t)HID * OUT) || load(argv[4], b2, OUT) ||
        load(argv[5], x, (size_t)1000 * IN) || runtime_init(r, w1, b1, w2, b2)) {
        fprintf(stderr, "model/runtime load failed\n");
        return 3;
    }
    FILE *f = fopen(argv[6], "rb");
    if (!f || fread(y, 1, 1000, f) != 1000) { fprintf(stderr, "label load failed\n"); return 3; }
    fclose(f);
    double sum = 0;
    unsigned correct = 0, checks = skip ? 1 : 1000;
    for (unsigned n = 0; n < checks; n++) {
        infer(r, x + (size_t)n * IN, out);
        int p = 0;
        for (uint32_t k = 0; k < OUT; k++) { sum += out[k]; if (out[k] > out[p]) p = k; }
        correct += p == y[n];
    }
    for (unsigned q = 0; q < reps; q++) {
        uint64_t t = mach_absolute_time();
        infer(r, x + (size_t)(q % 1000) * IN, out);
        dt[q] = mach_absolute_time() - t;
    }
    qsort(dt, reps, sizeof *dt, cmp64);
    volatile double sink = sum;
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    printf("engine\tnative\nwarm_median_ns\t%.3f\naccuracy\t%.6f\npeak_rss_bytes\t%ld\nchecksum\t%.9g\nparam_count\t%zu\n",
           ns(dt[reps / 2]), correct / (double)checks, (long)u.ru_maxrss, sink,
           (size_t)IN * HID + HID + (size_t)HID * OUT + OUT);
    return 0;
}
