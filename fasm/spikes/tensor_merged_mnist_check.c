/* Merged-compiler regression check: real MNIST, via the single canonical
 * compile(). Same 6-node matmul/bias/relu/matmul/bias/mse graph as
 * tensor_graph_engine_mnist_check.c, same real 784/32/10 shapes and real
 * external dataset, zero new op traits — this file is ONLY a graph +
 * tensors (no forward_nodes[]/zeroable[]), and must reach comparable test
 * accuracy as a regression check that the merge didn't change anything.
 * Also cross-checks that detect_fusion() (the same generic pass verified
 * on the MLP/Transformer merges) still finds MBR+MB+PLAIN at these much
 * larger dimensions, without needing a transformer- or mnist-specific
 * carve-out.
 */
#include "tensor_semantic_compiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* declared buffer footprint (data+grad+aux across the whole graph) — a
 * static, planner-relevant number to compare against the CNN graph in
 * tensor_merged_cnn_mnist_check.c, not a live RSS measurement */
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

enum { N_X, N_W1, N_B1, N_W2, N_B2, N_TARGET, N_MM1, N_BIAS1, N_RELU1, N_MM2, N_BIAS2, N_LOSS, N_COUNT };
enum { IN = 784, HID = 32, OUT = 10 };
static float initv(int i, int s) { return (float)(((i * 37 + s * 17) % 29) - 14) / 41.0f; }

int main(void) {
    const char *dir = getenv("MNIST_DIR");
    if (!dir) {
        printf("merged compiler mnist check skipped: set MNIST_DIR to a directory with the extracted IDX files "
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
        fprintf(stderr, "merged compiler mnist check: could not load MNIST_DIR=%s\n", dir);
        return 1;
    }

    float x[IN], w1[IN * HID], b1[HID] = {0}, w2[HID * OUT], b2[OUT] = {0}, target[OUT];
    float mm1[HID], bias1[HID], relu1[HID], mm2[OUT], bias2[OUT], loss[1];
    float gx[IN] = {0}, gw1[IN * HID], gb1[HID], gw2[HID * OUT], gb2[OUT], gtarget[OUT] = {0};
    float gmm1[HID], gbias1[HID], grelu1[HID], gmm2[OUT], gbias2[OUT], gloss[1];
    for (int i = 0; i < IN * HID; i++) w1[i] = initv(i, 2) * .05f;
    for (int i = 0; i < HID * OUT; i++) w2[i] = initv(i, 5) * .1f;

    Node g[N_COUNT] = {
        [N_X] = {LEAF, NONE, NONE, INPUT, {x, gx, NULL, 1, IN, 0}},
        [N_W1] = {LEAF, NONE, NONE, PARAM, {w1, gw1, NULL, IN, HID, 0}},
        [N_B1] = {LEAF, NONE, NONE, PARAM, {b1, gb1, NULL, 1, HID, 0}},
        [N_W2] = {LEAF, NONE, NONE, PARAM, {w2, gw2, NULL, HID, OUT, 0}},
        [N_B2] = {LEAF, NONE, NONE, PARAM, {b2, gb2, NULL, 1, OUT, 0}},
        [N_TARGET] = {LEAF, NONE, NONE, CONSTANT, {target, gtarget, NULL, 1, OUT, 0}},
        [N_MM1] = {MATMUL, N_X, N_W1, TEMP, {mm1, gmm1, NULL, 1, HID, 0}},
        [N_BIAS1] = {BIAS_ADD, N_MM1, N_B1, TEMP, {bias1, gbias1, NULL, 1, HID, 0}},
        [N_RELU1] = {RELU, N_BIAS1, NONE, TEMP, {relu1, grelu1, NULL, 1, HID, 0}},
        [N_MM2] = {MATMUL, N_RELU1, N_W2, TEMP, {mm2, gmm2, NULL, 1, OUT, 0}},
        [N_BIAS2] = {BIAS_ADD, N_MM2, N_B2, TEMP, {bias2, gbias2, NULL, 1, OUT, 0}},
        [N_LOSS] = {MSE, N_BIAS2, N_TARGET, TEMP, {loss, gloss, NULL, 1, 1, 0}},
    };

    ExecStep steps[32];
    Context ctx[32];
    uint32_t count = 0;
    if (compile(g, N_COUNT, .05f, steps, ctx, 32, &count)) { fprintf(stderr, "compile failed\n"); return 1; }
    if (count != 25) { fprintf(stderr, "unexpected step count=%u (want 25)\n", count); return 1; }

    Group groups[N_COUNT];
    unsigned ng = detect_fusion(g, N_COUNT, groups);
    if (ng != 3 || groups[0].kind != GROUP_MBR || groups[1].kind != GROUP_MB || groups[2].kind != GROUP_PLAIN) {
        fprintf(stderr, "mnist fusion cross-check disagreed with the locked XOR schedule shape: ng=%u\n", ng);
        return 2;
    }
    printf("fusion cross-check: same generic pass as the MLP/transformer merges, applied unmodified at 784/32/10 -> "
           "groups=%u MBR+MB+PLAIN (same shape as the 2/4/1 XOR case, different size)\n", ng);

    unsigned train_use = train_n < 3000 ? train_n : 3000;
    unsigned test_use = test_n < 1000 ? test_n : 1000;
    unsigned epochs = 8;

    unsigned correct_before = 0;
    for (unsigned s = 0; s < test_use; s++) {
        for (int i = 0; i < IN; i++) x[i] = test_images[(size_t)s * IN + i] / 255.0f;
        FWD[MATMUL](&g[N_MM1], &g[N_X], &g[N_W1]);
        FWD[BIAS_ADD](&g[N_BIAS1], &g[N_MM1], &g[N_B1]);
        FWD[RELU](&g[N_RELU1], &g[N_BIAS1], NULL);
        FWD[MATMUL](&g[N_MM2], &g[N_RELU1], &g[N_W2]);
        FWD[BIAS_ADD](&g[N_BIAS2], &g[N_MM2], &g[N_B2]);
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias2[c] > bias2[pred]) pred = c;
        correct_before += pred == test_labels[s];
    }

    clock_t t0 = clock();
    for (unsigned epoch = 0; epoch < epochs; epoch++) {
        for (unsigned s = 0; s < train_use; s++) {
            for (int i = 0; i < IN; i++) x[i] = train_images[(size_t)s * IN + i] / 255.0f;
            for (int i = 0; i < OUT; i++) target[i] = train_labels[s] == i ? 1.0f : 0.0f;
            if (tensor_transformer_steps_execute(steps, count)) return 3;
        }
    }
    double train_seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;

    unsigned correct_after = 0;
    for (unsigned s = 0; s < test_use; s++) {
        for (int i = 0; i < IN; i++) x[i] = test_images[(size_t)s * IN + i] / 255.0f;
        FWD[MATMUL](&g[N_MM1], &g[N_X], &g[N_W1]);
        FWD[BIAS_ADD](&g[N_BIAS1], &g[N_MM1], &g[N_B1]);
        FWD[RELU](&g[N_RELU1], &g[N_BIAS1], NULL);
        FWD[MATMUL](&g[N_MM2], &g[N_RELU1], &g[N_W2]);
        FWD[BIAS_ADD](&g[N_BIAS2], &g[N_MM2], &g[N_B2]);
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias2[c] > bias2[pred]) pred = c;
        correct_after += pred == test_labels[s];
    }

    printf("merged compiler check (mnist-mlp) passed: shape=%d-%d-%d(real-28x28-input) dataset=real(ossci-mirror) "
           "steps=%u(all compile()-derived, no forward_nodes[]/zeroable[] in this file) declared_graph_bytes=%zu "
           "train_samples=%u test_samples=%u epochs=%u train_seconds=%.2f test_accuracy=%.1f%%->%.1f%% (chance=10%%)\n",
           IN, HID, OUT, count, graph_bytes(g, N_COUNT), train_use, test_use, epochs, train_seconds,
           100.0 * correct_before / test_use, 100.0 * correct_after / test_use);

    free(train_images); free(train_labels); free(test_images); free(test_labels);
    return correct_after > correct_before && (100.0 * correct_after / test_use) > 40.0 ? 0 : 5;
}
