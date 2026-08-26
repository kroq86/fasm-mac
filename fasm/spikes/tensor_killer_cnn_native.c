/* Native arm64 deployment runner for the killer-workload's second model
 * type: a compact CNN (conv 5x5x4 -> relu -> 2x2 max pool -> fc 576->10),
 * real MNIST, batch=1. Same CLI shape and output format as
 * tensor_killer_mnist_native.c (w1 b1 w2 b2 inputs labels reps) on
 * purpose, so tensor_killer_compare.py and tensor_killer_lifetime_matrix.py
 * work against this model completely unchanged — only the graph inside
 * runtime_init() differs. Reuses CONV/POOL from tensor_semantic_compiler.h
 * exactly as already verified (tensor_merged_cnn_pool_mnist_check.c); no
 * compiler or kernel changes. Inference-only.
 *
 * w1=conv weight [COUT,CIN*KH*KW]. b1 is an unused placeholder (COUT zero
 * floats) kept only so the file-interface matches the MLP killer runner's
 * — this design has no conv bias, same reason the canonical compiler's
 * CONV kernel itself has none (see tensor_semantic_compiler.h). w2=fc
 * weight [POOL_OUT,OUT], b2=real fc bias [OUT].
 */
#define HIN 28
#define WIN 28
#define CIN 1
#define COUT 4
#define KH 5
#define KW 5
#define PHIN HOUT
#define PWIN WOUT
#define PCIN COUT
#define PPH 2
#define PPW 2
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include "tensor_semantic_compiler.h"

enum { CONV_OUT = HOUT * WOUT * COUT, POOL_OUT = PHOUT * PWOUT * PCIN, OUT = 10 };
enum { NX, NCW, NB1_UNUSED, NFCW, NFCB, NCONV, NRELU, NPOOL, NFC, NBIAS, NNODES };
int tensor_transformer_steps_execute(const ExecStep *s, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) { int rc = s[i].run(s[i].context); if (rc) return rc; }
    return 0;
}
typedef struct {
    float x[HIN * WIN * CIN], conv[CONV_OUT], relu[CONV_OUT], pool[POOL_OUT], pool_aux[POOL_OUT], fc[OUT], out[OUT];
    float gx[HIN * WIN * CIN], gcw[COUT * CIN * KH * KW], gfcw[POOL_OUT * OUT], gfcb[OUT];
    float gconv[CONV_OUT], grelu[CONV_OUT], gpool[POOL_OUT], gfc[OUT], gout[OUT];
    float dummy_b1, gdummy_b1; /* NB1_UNUSED's own buffer — must not alias fcb/gfcb */
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
static int runtime_init(Runtime *r, float *cw, float *fcw, float *fcb) {
    Node g[NNODES] = {
        [NX] = {LEAF, NONE, NONE, INPUT, {r->x, r->gx, NULL, 1, HIN * WIN * CIN, 0}},
        [NCW] = {LEAF, NONE, NONE, PARAM, {cw, r->gcw, NULL, COUT, CIN * KH * KW, 0}},
        [NB1_UNUSED] = {LEAF, NONE, NONE, PARAM, {&r->dummy_b1, &r->gdummy_b1, NULL, 1, 1, 0}},
        [NFCW] = {LEAF, NONE, NONE, PARAM, {fcw, r->gfcw, NULL, POOL_OUT, OUT, 0}},
        [NFCB] = {LEAF, NONE, NONE, PARAM, {fcb, r->gfcb, NULL, 1, OUT, 0}},
        [NCONV] = {CONV, NX, NCW, TEMP, {r->conv, r->gconv, NULL, 1, CONV_OUT, 0}},
        [NRELU] = {RELU, NCONV, NONE, TEMP, {r->relu, r->grelu, NULL, 1, CONV_OUT, 0}},
        [NPOOL] = {POOL, NRELU, NONE, TEMP, {r->pool, r->gpool, r->pool_aux, 1, POOL_OUT, POOL_OUT}},
        [NFC] = {MATMUL, NPOOL, NFCW, TEMP, {r->fc, r->gfc, NULL, 1, OUT, 0}},
        [NBIAS] = {BIAS_ADD, NFC, NFCB, TEMP, {r->out, r->gout, NULL, 1, OUT, 0}},
    };
    memcpy(r->g, g, sizeof g);
    /* NB1_UNUSED (index 2) would break compile()'s "lhs/rhs < i" ordering
     * check if any node referenced it — nothing does, it's a dead PARAM
     * leaf kept only to consume the b1.bin file. compile() still walks it
     * (LEAF, so no forward/backward step) and emits one harmless optimizer
     * step for it; that's fine, it's inference-only here so OPTIMIZER
     * steps are simply never executed. */
    return compile(r->g, NNODES, 0, r->steps, r->ctx, 32, &r->count) || r->count != 23;
}
static void infer(Runtime *r, const float *x, float *out) {
    memcpy(r->x, x, HIN * WIN * CIN * sizeof(float));
    /* forward-only slice: compile() emits forward(5) first for this graph
     * (conv,relu,pool,fc,bias), in node order — run exactly that. */
    if (tensor_transformer_steps_execute(r->steps, 5)) abort();
    memcpy(out, r->out, OUT * sizeof(float));
}
static double ns(uint64_t d) { mach_timebase_info_data_t t; mach_timebase_info(&t); return (double)d * t.numer / t.denom; }
static int cmp64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    if (argc != 8) { fprintf(stderr, "usage: %s w1(conv) b1(unused) w2(fc) b2(fc_bias) inputs labels reps\n", argv[0]); return 2; }
    unsigned reps = (unsigned)strtoul(argv[7], 0, 10);
    int skip = getenv("KILLER_SKIP_ACCURACY") != NULL;
    float *cw = malloc(COUT * CIN * KH * KW * 4), *b1_unused = malloc(4);
    float *fcw = malloc((size_t)POOL_OUT * OUT * 4), *fcb = malloc(OUT * 4);
    float *x = malloc((size_t)1000 * HIN * WIN * CIN * 4);
    uint64_t *dt = malloc((size_t)reps * sizeof *dt);
    unsigned char *y = malloc(1000);
    Runtime *r = calloc(1, sizeof *r);
    if (!cw || !b1_unused || !fcw || !fcb || !x || !dt || !y || !r ||
        load(argv[1], cw, COUT * CIN * KH * KW) || load(argv[2], b1_unused, 1) ||
        load(argv[3], fcw, (size_t)POOL_OUT * OUT) || load(argv[4], fcb, OUT) ||
        load(argv[5], x, (size_t)1000 * HIN * WIN * CIN) || runtime_init(r, cw, fcw, fcb)) {
        fprintf(stderr, "model/runtime load failed\n");
        return 3;
    }
    FILE *f = fopen(argv[6], "rb");
    if (!f || fread(y, 1, 1000, f) != 1000) { fprintf(stderr, "label load failed\n"); return 3; }
    fclose(f);
    float out[OUT], sum = 0;
    unsigned correct = 0, checks = skip ? 1 : 1000;
    for (unsigned n = 0; n < checks; n++) {
        infer(r, x + (size_t)n * HIN * WIN * CIN, out);
        int p = 0;
        for (int k = 0; k < OUT; k++) { sum += out[k]; if (out[k] > out[p]) p = k; }
        correct += p == y[n];
    }
    for (unsigned q = 0; q < reps; q++) {
        uint64_t t = mach_absolute_time();
        infer(r, x + (size_t)(q % 1000) * HIN * WIN * CIN, out);
        dt[q] = mach_absolute_time() - t;
    }
    qsort(dt, reps, sizeof *dt, cmp64);
    volatile float sink = sum;
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    printf("engine\tnative\nwarm_median_ns\t%.3f\naccuracy\t%.6f\npeak_rss_bytes\t%ld\nchecksum\t%.9g\n",
           ns(dt[reps / 2]), correct / (double)checks, (long)u.ru_maxrss, (double)sink);
    return !skip && correct < 500 ? 1 : 0;
}
