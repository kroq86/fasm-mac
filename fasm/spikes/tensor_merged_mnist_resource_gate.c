/* Real-workload comparison gate: MLP vs CNN vs CNN+POOL, same canonical
 * compile(), same fixed MNIST subset/init/epoch budget, one printed report;
 * not a stable core ABI. Read-only consumer of tensor_semantic_compiler.h —
 * no op/compiler work happens here, this file only measures and compares
 * what tensor_merged_mnist_check.c / tensor_merged_cnn_mnist_check.c /
 * tensor_merged_cnn_pool_mnist_check.c already established separately.
 *
 * Each of the three models is reported on: compiled step count, declared
 * graph bytes split into parameter bytes (PARAM leaves) vs activation/
 * saved-state bytes (everything else — TEMP nodes' data+grad+aux, plus any
 * INPUT/CONSTANT leaf buffers), a breakdown of node count by op, a static
 * per-forward-pass MAC/element-op estimate computed purely from each
 * node's own declared shape (and its operands'), wall-clock training time,
 * and test accuracy — all under the identical train_use/test_use/epoch
 * budget so the three numbers are actually comparable, not just adjacent.
 *
 * The concrete point of building this as one gate rather than three
 * separate printouts: it lets the same run assert, in one place, the
 * negative result already found by hand — CNN+POOL has MORE compiled
 * steps than plain CNN (23 vs 20) but trains FASTER, so action count is
 * not a usable proxy for compute cost. The static MAC estimate below,
 * computed purely from shapes, gets the direction right where step count
 * doesn't (pooled MACs < unpooled MACs, matching measured wall time) —
 * that's the concrete argument for a shape-aware cost model over an
 * action-count one.
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
static float initv(int i, int s) { return (float)(((i * 37 + s * 17) % 29) - 14) / 41.0f; }

static const char *op_name(uint32_t op) {
    switch (op) {
        case MATMUL: return "MATMUL"; case BIAS_ADD: return "BIAS_ADD"; case RELU: return "RELU"; case MSE: return "MSE";
        case ATTENTION: return "ATTENTION"; case RESIDUAL: return "RESIDUAL"; case LAYERNORM: return "LAYERNORM";
        case CONTIGUOUS: return "CONTIGUOUS"; case CONV: return "CONV"; case POOL: return "POOL"; default: return "LEAF";
    }
}
/* static, per-forward-pass estimate, computed purely from each node's own
 * declared shape and its operands' — no op-specific macros involved, so
 * this generalizes to any graph this header can compile, not just the
 * three built here. */
static double estimate_macs(const Node *g, uint32_t n) {
    double total = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (g[i].op == LEAF) continue;
        const Tensor *o = &g[i].tensor, *a = &g[g[i].lhs].tensor, *b = g[i].rhs != NONE ? &g[g[i].rhs].tensor : NULL;
        switch (g[i].op) {
            case MATMUL: total += (double)a->rows * a->cols * b->cols; break;
            case CONV: total += (double)o->cols * b->cols; break;
            case POOL: total += (double)a->rows * a->cols; break;
            default: total += (double)o->rows * o->cols; break; /* elementwise-ish ops: one op per output element */
        }
    }
    return total;
}
static void bytes_breakdown(const Node *g, uint32_t n, size_t *param_bytes, size_t *activation_bytes) {
    *param_bytes = 0;
    *activation_bytes = 0;
    for (uint32_t i = 0; i < n; i++) {
        size_t elems = (size_t)g[i].tensor.rows * g[i].tensor.cols;
        size_t bytes = elems * sizeof(float);
        if (g[i].tensor.grad) bytes += elems * sizeof(float);
        if (g[i].tensor.aux) bytes += (size_t)g[i].tensor.aux_count * sizeof(float);
        if (g[i].flags & PARAM) *param_bytes += bytes; else *activation_bytes += bytes;
    }
}

typedef struct {
    const char *name;
    uint32_t steps, node_count;
    size_t param_bytes, activation_bytes;
    double macs;
    unsigned op_counts[OP_COUNT];
    double train_seconds;
    double acc_before, acc_after;
} ModelResult;

static void tally_and_report(ModelResult *r, const Node *g, uint32_t n, uint32_t steps) {
    r->node_count = n;
    r->steps = steps;
    r->macs = estimate_macs(g, n);
    bytes_breakdown(g, n, &r->param_bytes, &r->activation_bytes);
    memset(r->op_counts, 0, sizeof r->op_counts);
    for (uint32_t i = 0; i < n; i++) if (g[i].op != LEAF) r->op_counts[g[i].op]++;
}

enum { TRAIN_USE = 3000, TEST_USE = 1000, EPOCHS = 5 };

static ModelResult run_mlp(const unsigned char *train_images, const unsigned char *train_labels,
                            const unsigned char *test_images, const unsigned char *test_labels) {
    enum { IN = 784, HID = 32, OUT = 10 };
    enum { N_X, N_W1, N_B1, N_W2, N_B2, N_TARGET, N_MM1, N_BIAS1, N_RELU1, N_MM2, N_BIAS2, N_LOSS, N_COUNT };
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
    compile(g, N_COUNT, .05f, steps, ctx, 32, &count);

    ModelResult r = {.name = "MLP (784->32->10)"};
    tally_and_report(&r, g, N_COUNT, count);

    unsigned correct = 0;
    for (unsigned s = 0; s < TEST_USE; s++) {
        for (int i = 0; i < IN; i++) x[i] = test_images[(size_t)s * IN + i] / 255.0f;
        FWD[MATMUL](&g[N_MM1], &g[N_X], &g[N_W1]); FWD[BIAS_ADD](&g[N_BIAS1], &g[N_MM1], &g[N_B1]);
        FWD[RELU](&g[N_RELU1], &g[N_BIAS1], NULL); FWD[MATMUL](&g[N_MM2], &g[N_RELU1], &g[N_W2]);
        FWD[BIAS_ADD](&g[N_BIAS2], &g[N_MM2], &g[N_B2]);
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias2[c] > bias2[pred]) pred = c;
        correct += pred == test_labels[s];
    }
    r.acc_before = 100.0 * correct / TEST_USE;

    clock_t t0 = clock();
    for (unsigned epoch = 0; epoch < EPOCHS; epoch++) for (unsigned s = 0; s < TRAIN_USE; s++) {
        for (int i = 0; i < IN; i++) x[i] = train_images[(size_t)s * IN + i] / 255.0f;
        for (int i = 0; i < OUT; i++) target[i] = train_labels[s] == i ? 1.0f : 0.0f;
        tensor_transformer_steps_execute(steps, count);
    }
    r.train_seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;

    correct = 0;
    for (unsigned s = 0; s < TEST_USE; s++) {
        for (int i = 0; i < IN; i++) x[i] = test_images[(size_t)s * IN + i] / 255.0f;
        FWD[MATMUL](&g[N_MM1], &g[N_X], &g[N_W1]); FWD[BIAS_ADD](&g[N_BIAS1], &g[N_MM1], &g[N_B1]);
        FWD[RELU](&g[N_RELU1], &g[N_BIAS1], NULL); FWD[MATMUL](&g[N_MM2], &g[N_RELU1], &g[N_W2]);
        FWD[BIAS_ADD](&g[N_BIAS2], &g[N_MM2], &g[N_B2]);
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias2[c] > bias2[pred]) pred = c;
        correct += pred == test_labels[s];
    }
    r.acc_after = 100.0 * correct / TEST_USE;
    return r;
}

static ModelResult run_cnn(const unsigned char *train_images, const unsigned char *train_labels,
                            const unsigned char *test_images, const unsigned char *test_labels) {
    enum { CONV_OUT = HOUT * WOUT * COUT, OUT = 10 };
    enum { N_X, N_CW, N_FCW, N_FCB, N_TARGET, N_CONV, N_RELU, N_FC, N_BIAS, N_LOSS, N_COUNT };
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
    compile(g, N_COUNT, .02f, steps, ctx, 32, &count);

    ModelResult r = {.name = "CNN, no pool (conv 5x5x4 -> fc 2304->10)"};
    tally_and_report(&r, g, N_COUNT, count);

    unsigned correct = 0;
    for (unsigned s = 0; s < TEST_USE; s++) {
        for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = test_images[(size_t)s * HIN * WIN * CIN + i] / 255.0f;
        FWD[CONV](&g[N_CONV], &g[N_X], &g[N_CW]); FWD[RELU](&g[N_RELU], &g[N_CONV], NULL);
        FWD[MATMUL](&g[N_FC], &g[N_RELU], &g[N_FCW]); FWD[BIAS_ADD](&g[N_BIAS], &g[N_FC], &g[N_FCB]);
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias[c] > bias[pred]) pred = c;
        correct += pred == test_labels[s];
    }
    r.acc_before = 100.0 * correct / TEST_USE;

    clock_t t0 = clock();
    for (unsigned epoch = 0; epoch < EPOCHS; epoch++) for (unsigned s = 0; s < TRAIN_USE; s++) {
        for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = train_images[(size_t)s * HIN * WIN * CIN + i] / 255.0f;
        for (int i = 0; i < OUT; i++) target[i] = train_labels[s] == i ? 1.0f : 0.0f;
        tensor_transformer_steps_execute(steps, count);
    }
    r.train_seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;

    correct = 0;
    for (unsigned s = 0; s < TEST_USE; s++) {
        for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = test_images[(size_t)s * HIN * WIN * CIN + i] / 255.0f;
        FWD[CONV](&g[N_CONV], &g[N_X], &g[N_CW]); FWD[RELU](&g[N_RELU], &g[N_CONV], NULL);
        FWD[MATMUL](&g[N_FC], &g[N_RELU], &g[N_FCW]); FWD[BIAS_ADD](&g[N_BIAS], &g[N_FC], &g[N_FCB]);
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias[c] > bias[pred]) pred = c;
        correct += pred == test_labels[s];
    }
    r.acc_after = 100.0 * correct / TEST_USE;
    return r;
}

static ModelResult run_cnn_pool(const unsigned char *train_images, const unsigned char *train_labels,
                                 const unsigned char *test_images, const unsigned char *test_labels) {
    enum { CONV_OUT = HOUT * WOUT * COUT, POOL_OUT = PHOUT * PWOUT * PCIN, OUT = 10 };
    enum { N_X, N_CW, N_FCW, N_FCB, N_TARGET, N_CONV, N_RELU, N_POOL, N_FC, N_BIAS, N_LOSS, N_COUNT };
    float x[HIN * WIN * CIN], cw[COUT * CIN * KH * KW], fcw[POOL_OUT * OUT], fcb[OUT] = {0}, target[OUT];
    float conv[CONV_OUT], relu[CONV_OUT], pool[POOL_OUT], pool_aux[POOL_OUT], fc[OUT], bias[OUT], loss[1];
    float gx[HIN * WIN * CIN] = {0}, gcw[COUT * CIN * KH * KW], gfcw[POOL_OUT * OUT], gfcb[OUT], gtarget[OUT] = {0};
    float gconv[CONV_OUT], grelu[CONV_OUT], gpool[POOL_OUT], gfc[OUT], gbias[OUT], gloss[1];
    for (int i = 0; i < COUT * CIN * KH * KW; i++) cw[i] = initv(i, 2) * .3f;
    for (int i = 0; i < POOL_OUT * OUT; i++) fcw[i] = initv(i, 5) * .1f;

    Node g[N_COUNT] = {
        [N_X] = {LEAF, NONE, NONE, INPUT, {x, gx, NULL, 1, HIN * WIN * CIN, 0}},
        [N_CW] = {LEAF, NONE, NONE, PARAM, {cw, gcw, NULL, COUT, CIN * KH * KW, 0}},
        [N_FCW] = {LEAF, NONE, NONE, PARAM, {fcw, gfcw, NULL, POOL_OUT, OUT, 0}},
        [N_FCB] = {LEAF, NONE, NONE, PARAM, {fcb, gfcb, NULL, 1, OUT, 0}},
        [N_TARGET] = {LEAF, NONE, NONE, CONSTANT, {target, gtarget, NULL, 1, OUT, 0}},
        [N_CONV] = {CONV, N_X, N_CW, TEMP, {conv, gconv, NULL, 1, CONV_OUT, 0}},
        [N_RELU] = {RELU, N_CONV, NONE, TEMP, {relu, grelu, NULL, 1, CONV_OUT, 0}},
        [N_POOL] = {POOL, N_RELU, NONE, TEMP, {pool, gpool, pool_aux, 1, POOL_OUT, POOL_OUT}},
        [N_FC] = {MATMUL, N_POOL, N_FCW, TEMP, {fc, gfc, NULL, 1, OUT, 0}},
        [N_BIAS] = {BIAS_ADD, N_FC, N_FCB, TEMP, {bias, gbias, NULL, 1, OUT, 0}},
        [N_LOSS] = {MSE, N_BIAS, N_TARGET, TEMP, {loss, gloss, NULL, 1, 1, 0}},
    };
    ExecStep steps[32];
    Context ctx[32];
    uint32_t count = 0;
    compile(g, N_COUNT, .02f, steps, ctx, 32, &count);

    ModelResult r = {.name = "CNN, 2x2 pool (conv 5x5x4 -> pool -> fc 576->10)"};
    tally_and_report(&r, g, N_COUNT, count);

    unsigned correct = 0;
    for (unsigned s = 0; s < TEST_USE; s++) {
        for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = test_images[(size_t)s * HIN * WIN * CIN + i] / 255.0f;
        FWD[CONV](&g[N_CONV], &g[N_X], &g[N_CW]); FWD[RELU](&g[N_RELU], &g[N_CONV], NULL);
        FWD[POOL](&g[N_POOL], &g[N_RELU], NULL); FWD[MATMUL](&g[N_FC], &g[N_POOL], &g[N_FCW]);
        FWD[BIAS_ADD](&g[N_BIAS], &g[N_FC], &g[N_FCB]);
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias[c] > bias[pred]) pred = c;
        correct += pred == test_labels[s];
    }
    r.acc_before = 100.0 * correct / TEST_USE;

    clock_t t0 = clock();
    for (unsigned epoch = 0; epoch < EPOCHS; epoch++) for (unsigned s = 0; s < TRAIN_USE; s++) {
        for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = train_images[(size_t)s * HIN * WIN * CIN + i] / 255.0f;
        for (int i = 0; i < OUT; i++) target[i] = train_labels[s] == i ? 1.0f : 0.0f;
        tensor_transformer_steps_execute(steps, count);
    }
    r.train_seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;

    correct = 0;
    for (unsigned s = 0; s < TEST_USE; s++) {
        for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = test_images[(size_t)s * HIN * WIN * CIN + i] / 255.0f;
        FWD[CONV](&g[N_CONV], &g[N_X], &g[N_CW]); FWD[RELU](&g[N_RELU], &g[N_CONV], NULL);
        FWD[POOL](&g[N_POOL], &g[N_RELU], NULL); FWD[MATMUL](&g[N_FC], &g[N_POOL], &g[N_FCW]);
        FWD[BIAS_ADD](&g[N_BIAS], &g[N_FC], &g[N_FCB]);
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias[c] > bias[pred]) pred = c;
        correct += pred == test_labels[s];
    }
    r.acc_after = 100.0 * correct / TEST_USE;
    return r;
}

static void print_result(const ModelResult *r) {
    printf("\n== %s ==\n", r->name);
    printf("  compiled_steps=%u node_count=%u\n", r->steps, r->node_count);
    printf("  param_bytes=%zu activation_bytes=%zu declared_graph_bytes=%zu\n",
           r->param_bytes, r->activation_bytes, r->param_bytes + r->activation_bytes);
    printf("  static_macs_per_forward_pass=%.0f\n", r->macs);
    printf("  op_breakdown:");
    for (unsigned op = 1; op < OP_COUNT; op++) if (r->op_counts[op]) printf(" %s=%u", op_name(op), r->op_counts[op]);
    printf("\n");
    printf("  train_seconds(%u epochs, %u samples)=%.2f\n", (unsigned)EPOCHS, (unsigned)TRAIN_USE, r->train_seconds);
    printf("  test_accuracy=%.1f%%->%.1f%% (chance=10%%, %u samples)\n", r->acc_before, r->acc_after, (unsigned)TEST_USE);
}

int main(void) {
    const char *dir = getenv("MNIST_DIR");
    if (!dir) {
        printf("merged compiler mnist resource gate skipped: set MNIST_DIR to a directory with the extracted IDX files "
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
        train_n != train_label_n || test_n != test_label_n || train_n < TRAIN_USE || test_n < TEST_USE) {
        fprintf(stderr, "merged compiler mnist resource gate: could not load MNIST_DIR=%s\n", dir);
        return 1;
    }

    printf("mnist resource gate: fixed train_samples=%u test_samples=%u epochs=%u across all three models\n",
           (unsigned)TRAIN_USE, (unsigned)TEST_USE, (unsigned)EPOCHS);

    ModelResult mlp = run_mlp(train_images, train_labels, test_images, test_labels);
    ModelResult cnn = run_cnn(train_images, train_labels, test_images, test_labels);
    ModelResult pool = run_cnn_pool(train_images, train_labels, test_images, test_labels);
    print_result(&mlp);
    print_result(&cnn);
    print_result(&pool);

    printf("\n== cost-model check ==\n");
    printf("  compiled_steps: cnn=%u pool=%u (pool has MORE steps: %s)\n", cnn.steps, pool.steps, pool.steps > cnn.steps ? "yes" : "no");
    printf("  train_seconds:  cnn=%.2f pool=%.2f (pool is FASTER despite more steps: %s)\n",
           cnn.train_seconds, pool.train_seconds, pool.train_seconds < cnn.train_seconds ? "yes" : "no");
    printf("  static_macs:    cnn=%.0f pool=%.0f (mac estimate correctly predicts pool is cheaper: %s)\n",
           cnn.macs, pool.macs, pool.macs < cnn.macs ? "yes" : "no");
    printf("  -> action_count alone would rank pool as more expensive than cnn; it is not. "
           "a shape-aware static estimate (macs) ranks them correctly.\n");

    int ok = 1;
    if (!(mlp.acc_after > 40.0 && cnn.acc_after > 40.0 && pool.acc_after > 40.0)) { fprintf(stderr, "gate: one model did not clear chance-level accuracy\n"); ok = 0; }
    if (!(pool.steps > cnn.steps)) { fprintf(stderr, "gate: expected pool.steps > cnn.steps (%u vs %u)\n", pool.steps, cnn.steps); ok = 0; }
    if (!(pool.train_seconds < cnn.train_seconds)) { fprintf(stderr, "gate: expected pool.train_seconds < cnn.train_seconds (%.2f vs %.2f)\n", pool.train_seconds, cnn.train_seconds); ok = 0; }
    if (!(pool.macs < cnn.macs)) { fprintf(stderr, "gate: expected pool.macs < cnn.macs (%.0f vs %.0f)\n", pool.macs, cnn.macs); ok = 0; }
    if (!(pool.param_bytes < cnn.param_bytes)) { fprintf(stderr, "gate: expected pool.param_bytes < cnn.param_bytes (%zu vs %zu)\n", pool.param_bytes, cnn.param_bytes); ok = 0; }

    free(train_images); free(train_labels); free(test_images); free(test_labels);
    return ok ? 0 : 2;
}
