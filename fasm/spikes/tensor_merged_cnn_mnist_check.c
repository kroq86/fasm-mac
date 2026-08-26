/* Merged-compiler generality check: CONV on real MNIST, through the same
 * canonical compile() and the same shared executor as every other merged
 * check; not a stable core ABI.
 *
 * tensor_merged_cnn_check.c proved the architectural thesis on a synthetic
 * fixture: adding CONV touched the op table, not compile()/validate() for
 * any pre-existing op. This is the scale test: the same real 28x28 MNIST
 * digits tensor_merged_mnist_check.c already trains an MLP on, this time
 * through conv(5x5,4 filters)->relu->FC->bias->mse. No flatten/view op was
 * needed — CONV's output is already declared as a flat [1,HOUT*WOUT*COUT]
 * node (tensor_semantic_compiler.h), so the very next MATMUL consumes it
 * directly; the "reshape" is a free reinterpretation of the same memory,
 * not a new op trait or an MNIST-specific carve-out.
 *
 * Comparable to tensor_merged_mnist_check.c on every axis the two are
 * asked to be compared on: compiled step count, declared graph memory
 * footprint (data+grad+aux across the whole graph), wall-clock training
 * time, and test accuracy on the same train/test subset sizes.
 */
#define HIN 28
#define WIN 28
#define CIN 1
#define COUT 4
#define KH 5
#define KW 5
#include "tensor_semantic_compiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t read_be32(FILE *f) { unsigned char b[4]; if (fread(b, 1, 4, f) != 4) return 0; return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3]; }
static unsigned char *read_idx_images(const char *path, uint32_t *count, uint32_t *rows, uint32_t *cols) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic = read_be32(f);
    *count = read_be32(f); *rows = read_be32(f); *cols = read_be32(f);
    if (magic != 0x00000803 || !*count || !*rows || !*cols) { fclose(f); return NULL; }
    size_t n = (size_t)*count * *rows * *cols;
    unsigned char *buf = malloc(n);
    if (!buf || fread(buf, 1, n, f) != n) { free(buf); fclose(f); return NULL; }
    fclose(f);
    return buf;
}
static unsigned char *read_idx_labels(const char *path, uint32_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic = read_be32(f);
    *count = read_be32(f);
    if (magic != 0x00000801 || !*count) { fclose(f); return NULL; }
    unsigned char *buf = malloc(*count);
    if (!buf || fread(buf, 1, *count, f) != *count) { free(buf); fclose(f); return NULL; }
    fclose(f);
    return buf;
}
static size_t graph_bytes(const Node *g, uint32_t n) {
    size_t bytes = 0;
    for (uint32_t i = 0; i < n; i++) {
        size_t elems = (size_t)g[i].tensor.rows * g[i].tensor.cols;
        bytes += elems * sizeof(float);
        if (g[i].tensor.grad) bytes += elems * sizeof(float);
        if (g[i].tensor.aux) bytes += (size_t)g[i].tensor.aux_count * sizeof(float);
    }
    return bytes;
}

enum { CONV_OUT = HOUT * WOUT * COUT }; /* 24*24*4 = 2304 */
enum { OUT = 10 };
enum { N_X, N_CW, N_FCW, N_FCB, N_TARGET, N_CONV, N_RELU, N_FC, N_BIAS, N_LOSS, N_COUNT };
static float initv(int i, int s) { return (float)(((i * 37 + s * 17) % 29) - 14) / 41.0f; }

int main(void) {
    const char *dir = getenv("MNIST_DIR");
    if (!dir) {
        printf("merged compiler cnn-mnist check skipped: set MNIST_DIR to a directory with the extracted IDX files "
               "(https://ossci-datasets.s3.amazonaws.com/mnist/)\n");
        return 0;
    }
    char path[1024];
    snprintf(path, sizeof path, "%s/train-images-idx3-ubyte", dir);
    uint32_t train_n, rows, cols;
    unsigned char *train_images = read_idx_images(path, &train_n, &rows, &cols);
    snprintf(path, sizeof path, "%s/train-labels-idx1-ubyte", dir);
    uint32_t train_label_n;
    unsigned char *train_labels = read_idx_labels(path, &train_label_n);
    snprintf(path, sizeof path, "%s/t10k-images-idx3-ubyte", dir);
    uint32_t test_n, test_rows, test_cols;
    unsigned char *test_images = read_idx_images(path, &test_n, &test_rows, &test_cols);
    snprintf(path, sizeof path, "%s/t10k-labels-idx1-ubyte", dir);
    uint32_t test_label_n;
    unsigned char *test_labels = read_idx_labels(path, &test_label_n);
    if (!train_images || !train_labels || !test_images || !test_labels || rows != 28 || cols != 28 ||
        train_n != train_label_n || test_n != test_label_n) {
        fprintf(stderr, "merged compiler cnn-mnist check: could not load MNIST_DIR=%s\n", dir);
        return 1;
    }

    float x[HIN * WIN * CIN], cw[COUT * CIN * KH * KW], fcw[CONV_OUT * OUT], fcb[OUT] = {0}, target[OUT];
    float conv[CONV_OUT], relu[CONV_OUT], fc[OUT], bias[OUT], loss[1];
    float gx[HIN * WIN * CIN] = {0}, gcw[COUT * CIN * KH * KW], gfcw[CONV_OUT * OUT], gfcb[OUT], gtarget[OUT] = {0};
    float gconv[CONV_OUT], grelu[CONV_OUT], gfc[OUT], gbias[OUT], gloss[1];
    for (int i = 0; i < COUT * CIN * KH * KW; i++) cw[i] = initv(i, 2) * .3f;
    for (int i = 0; i < CONV_OUT * OUT; i++) fcw[i] = initv(i, 5) * .05f;

    Node g[N_COUNT] = {
        [N_X] = {LEAF, NONE, NONE, INPUT, {x, gx, NULL, 1, HIN * WIN * CIN, 0}},
        [N_CW] = {LEAF, NONE, NONE, PARAM, {cw, gcw, NULL, COUT, CIN * KH * KW, 0}},
        [N_FCW] = {LEAF, NONE, NONE, PARAM, {fcw, gfcw, NULL, CONV_OUT, OUT, 0}},
        [N_FCB] = {LEAF, NONE, NONE, PARAM, {fcb, gfcb, NULL, 1, OUT, 0}},
        [N_TARGET] = {LEAF, NONE, NONE, CONSTANT, {target, gtarget, NULL, 1, OUT, 0}},
        [N_CONV] = {CONV, N_X, N_CW, TEMP, {conv, gconv, NULL, 1, CONV_OUT, 0}},
        [N_RELU] = {RELU, N_CONV, NONE, TEMP, {relu, grelu, NULL, 1, CONV_OUT, 0}},
        [N_FC] = {MATMUL, N_RELU, N_FCW, TEMP, {fc, gfc, NULL, 1, OUT, 0}},
        [N_BIAS] = {BIAS_ADD, N_FC, N_FCB, TEMP, {bias, gbias, NULL, 1, OUT, 0}},
        [N_LOSS] = {MSE, N_BIAS, N_TARGET, TEMP, {loss, gloss, NULL, 1, 1, 0}},
    };

    ExecStep steps[32];
    Context ctx[32];
    uint32_t count = 0;
    if (compile(g, N_COUNT, .02f, steps, ctx, 32, &count)) { fprintf(stderr, "compile failed\n"); return 1; }
    if (count != 20) { fprintf(stderr, "unexpected step count=%u (want 20)\n", count); return 1; }

    /* finite differences on a small fixture (the first test image) before
     * any real training touches the weights — both the new conv weight and
     * the pre-existing FC weight, same pattern as tensor_merged_cnn_check.c */
    for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = test_images[i] / 255.0f;
    for (int i = 0; i < OUT; i++) target[i] = test_labels[0] == i ? 1.0f : 0.0f;
    if (tensor_transformer_steps_execute(steps, 5 + 7 + 5)) return 2; /* forward+zero+backward, no optimizer */
    struct { float *param, *grad; int idx; const char *name; } probes[] = {
        {cw, gcw, 3, "conv_weight"}, {fcw, gfcw, 12, "fc_weight"},
    };
    for (unsigned p = 0; p < 2; p++) {
        float eps = 1e-2f, old = probes[p].param[probes[p].idx], analytic = probes[p].grad[probes[p].idx];
        probes[p].param[probes[p].idx] = old + eps;
        if (tensor_transformer_steps_execute(steps, 5)) return 2;
        float plus = loss[0];
        probes[p].param[probes[p].idx] = old - eps;
        if (tensor_transformer_steps_execute(steps, 5)) return 2;
        float minus = loss[0];
        probes[p].param[probes[p].idx] = old;
        float numeric = (plus - minus) / (2 * eps);
        if (fabsf(numeric - analytic) > 5e-2f * fmaxf(1, fmaxf(fabsf(numeric), fabsf(analytic)))) {
            fprintf(stderr, "merged compiler cnn-mnist backward mismatch on %s: analytic=%g numeric=%g\n", probes[p].name, analytic, numeric);
            return 2;
        }
    }
    for (int i = 0; i < COUT * CIN * KH * KW; i++) cw[i] = initv(i, 2) * .3f;
    for (int i = 0; i < CONV_OUT * OUT; i++) fcw[i] = initv(i, 5) * .05f;
    memset(fcb, 0, sizeof fcb);
    printf("finite-difference check passed on real MNIST fixture: conv_weight and fc_weight both match\n");

    unsigned train_use = train_n < 3000 ? train_n : 3000;
    unsigned test_use = test_n < 1000 ? test_n : 1000;
    unsigned epochs = 5;

    unsigned correct_before = 0;
    for (unsigned s = 0; s < test_use; s++) {
        for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = test_images[(size_t)s * HIN * WIN * CIN + i] / 255.0f;
        FWD[CONV](&g[N_CONV], &g[N_X], &g[N_CW]);
        FWD[RELU](&g[N_RELU], &g[N_CONV], NULL);
        FWD[MATMUL](&g[N_FC], &g[N_RELU], &g[N_FCW]);
        FWD[BIAS_ADD](&g[N_BIAS], &g[N_FC], &g[N_FCB]);
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias[c] > bias[pred]) pred = c;
        correct_before += pred == test_labels[s];
    }

    clock_t t0 = clock();
    for (unsigned epoch = 0; epoch < epochs; epoch++) {
        for (unsigned s = 0; s < train_use; s++) {
            for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = train_images[(size_t)s * HIN * WIN * CIN + i] / 255.0f;
            for (int i = 0; i < OUT; i++) target[i] = train_labels[s] == i ? 1.0f : 0.0f;
            if (tensor_transformer_steps_execute(steps, count)) return 3;
        }
    }
    double train_seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;

    unsigned correct_after = 0;
    for (unsigned s = 0; s < test_use; s++) {
        for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = test_images[(size_t)s * HIN * WIN * CIN + i] / 255.0f;
        FWD[CONV](&g[N_CONV], &g[N_X], &g[N_CW]);
        FWD[RELU](&g[N_RELU], &g[N_CONV], NULL);
        FWD[MATMUL](&g[N_FC], &g[N_RELU], &g[N_FCW]);
        FWD[BIAS_ADD](&g[N_BIAS], &g[N_FC], &g[N_FCB]);
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias[c] > bias[pred]) pred = c;
        correct_after += pred == test_labels[s];
    }

    printf("merged compiler check (mnist-cnn) passed: shape=conv(%dx%dx%d->%dx%dx%d)->fc(%d->%d) "
           "dataset=real(ossci-mirror) steps=%u(all compile()-derived, no forward_nodes[]/zeroable[]/flatten-op "
           "in this file) declared_graph_bytes=%zu train_samples=%u test_samples=%u epochs=%u train_seconds=%.2f "
           "test_accuracy=%.1f%%->%.1f%% (chance=10%%)\n",
           HIN, WIN, CIN, HOUT, WOUT, COUT, CONV_OUT, OUT, count, graph_bytes(g, N_COUNT), train_use, test_use, epochs,
           train_seconds, 100.0 * correct_before / test_use, 100.0 * correct_after / test_use);

    free(train_images); free(train_labels); free(test_images); free(test_labels);
    return correct_after > correct_before && (100.0 * correct_after / test_use) > 40.0 ? 0 : 5;
}
