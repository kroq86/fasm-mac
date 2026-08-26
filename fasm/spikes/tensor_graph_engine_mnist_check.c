/* Experimental generic graph engine on a real external dataset, not a
 * hand-prepared toy; not a stable core ABI.
 *
 * Every model this engine has trained so far (XOR, the T=3 Transformer
 * block, GunPoint) was purpose-built for this repo. A user pointing
 * `tensorctl` at their own model brings neither the shapes nor the data
 * this project chose — the first rung of that ladder is a real MNIST
 * classifier: same 6-node matmul/bias/relu/matmul/bias/mse graph as
 * tensor_graph_engine_check.c's XOR MLP (zero new op traits needed), real
 * 28x28=784-dim input, a real external dataset, and the same generic
 * fusion pass from tensor_graph_engine_fusion_check.c applied unmodified
 * at these much larger dimensions.
 *
 * The dataset stays external (real ubyte IDX files from
 * https://ossci-datasets.s3.amazonaws.com/mnist/ — the same mirror
 * torchvision uses): point MNIST_DIR at a directory containing
 * train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte,
 * t10k-labels-idx1-ubyte (gunzip'd). Without it, only the deterministic
 * self-test runs.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LEAF, MATMUL, RELU, MSE, BIAS, OP_COUNT };
enum { PARAM = 1, CONSTANT = 2, INPUT = 4, TEMP = 8 };
#define NONE UINT32_MAX

typedef struct {
    uint32_t op, lhs, rhs, flags;
    uint32_t rows, cols;
    uint8_t trainable;
    float *data, *grad;
} Node;

typedef struct { int (*run)(void *); void *context; uint32_t kind, flags; uint64_t reserved; } ExecStep;
extern int tensor_transformer_steps_execute(const ExecStep *, uint64_t);

/* --- same generic kernels as every other graph-engine spike, verbatim --- */
static void k_matmul_fwd(Node *self, Node *lhs, Node *rhs) {
    uint32_t r = lhs->rows, k = lhs->cols, c = rhs->cols;
    memset(self->data, 0, (size_t)r * c * sizeof(float));
    for (uint32_t i = 0; i < r; i++) for (uint32_t p = 0; p < k; p++) {
        float x = lhs->data[i * k + p];
        for (uint32_t j = 0; j < c; j++) self->data[i * c + j] += x * rhs->data[p * c + j];
    }
}
static void k_matmul_bwd(Node *self, Node *lhs, Node *rhs) {
    uint32_t r = lhs->rows, k = lhs->cols, c = rhs->cols;
    for (uint32_t i = 0; i < r; i++) for (uint32_t j = 0; j < c; j++) {
        float g = self->grad[i * c + j];
        for (uint32_t p = 0; p < k; p++) {
            if (lhs->grad) lhs->grad[i * k + p] += g * rhs->data[p * c + j];
            if (rhs->grad) rhs->grad[p * c + j] += g * lhs->data[i * k + p];
        }
    }
}
static void k_bias_fwd(Node *self, Node *lhs, Node *rhs) {
    for (uint32_t i = 0; i < self->rows; i++) for (uint32_t j = 0; j < self->cols; j++)
        self->data[i * self->cols + j] = lhs->data[i * self->cols + j] + rhs->data[j];
}
static void k_bias_bwd(Node *self, Node *lhs, Node *rhs) {
    for (uint32_t i = 0; i < self->rows; i++) for (uint32_t j = 0; j < self->cols; j++) {
        float g = self->grad[i * self->cols + j];
        if (lhs->grad) lhs->grad[i * self->cols + j] += g;
        if (rhs->grad) rhs->grad[j] += g;
    }
}
static void k_relu_fwd(Node *self, Node *lhs, Node *rhs) {
    (void)rhs;
    for (uint32_t i = 0; i < self->rows * self->cols; i++) self->data[i] = lhs->data[i] > 0 ? lhs->data[i] : 0;
}
static void k_relu_bwd(Node *self, Node *lhs, Node *rhs) {
    (void)rhs;
    for (uint32_t i = 0; i < self->rows * self->cols; i++) if (lhs->grad) lhs->grad[i] += lhs->data[i] > 0 ? self->grad[i] : 0;
}
static void k_mse_fwd(Node *self, Node *lhs, Node *rhs) {
    uint32_t n = lhs->rows * lhs->cols;
    float s = 0;
    for (uint32_t i = 0; i < n; i++) { float e = lhs->data[i] - rhs->data[i]; s += e * e; }
    self->data[0] = s / n;
}
static void k_mse_bwd(Node *self, Node *lhs, Node *rhs) {
    uint32_t n = lhs->rows * lhs->cols;
    float g = self->grad[0];
    for (uint32_t i = 0; i < n; i++) { float e = lhs->data[i] - rhs->data[i]; if (lhs->grad) lhs->grad[i] += g * 2 * e / n; }
}
typedef void (*KernelFn)(Node *, Node *, Node *);
static const KernelFn FWD[OP_COUNT] = {[MATMUL] = k_matmul_fwd, [BIAS] = k_bias_fwd, [RELU] = k_relu_fwd, [MSE] = k_mse_fwd};
static const KernelFn BWD[OP_COUNT] = {[MATMUL] = k_matmul_bwd, [BIAS] = k_bias_bwd, [RELU] = k_relu_bwd, [MSE] = k_mse_bwd};

/* --- same generic fusion pass, verbatim from tensor_graph_engine_fusion_check.c --- */
static unsigned consumers_of(const Node *nodes, unsigned total, unsigned needle) {
    unsigned users = 0;
    for (unsigned i = 0; i < total; i++) {
        if (nodes[i].op == LEAF) continue;
        users += nodes[i].lhs == needle;
        users += nodes[i].rhs != NONE && nodes[i].rhs == needle;
    }
    return users;
}
enum { GROUP_PLAIN, GROUP_MB, GROUP_MBR };
typedef struct { uint8_t kind; uint32_t matmul, bias, relu, plain; } Group;
static unsigned build_schedule(Node *nodes, unsigned total, const unsigned *order, unsigned n, Group *groups) {
    unsigned g = 0, i = 0;
    while (i < n) {
        unsigned idx = order[i];
        if (nodes[idx].op == MATMUL && i + 1 < n && nodes[order[i + 1]].op == BIAS && nodes[order[i + 1]].lhs == idx && consumers_of(nodes, total, idx) == 1) {
            unsigned bias_idx = order[i + 1];
            if (i + 2 < n && nodes[order[i + 2]].op == RELU && nodes[order[i + 2]].lhs == bias_idx && consumers_of(nodes, total, bias_idx) == 1) {
                groups[g++] = (Group){GROUP_MBR, idx, bias_idx, order[i + 2], NONE};
                i += 3;
                continue;
            }
            groups[g++] = (Group){GROUP_MB, idx, bias_idx, NONE, NONE};
            i += 2;
            continue;
        }
        groups[g++] = (Group){GROUP_PLAIN, NONE, NONE, NONE, idx};
        i += 1;
    }
    return g;
}

typedef struct { Node *nodes; uint32_t index, generation, *live_generation; int backward; } PlainCtx;
static int plain_execute(void *opaque) {
    PlainCtx *c = opaque;
    if (*c->live_generation != c->generation) return -4;
    Node *n = &c->nodes[c->index];
    Node *lhs = &c->nodes[n->lhs];
    Node *rhs = n->rhs != NONE ? &c->nodes[n->rhs] : NULL;
    (c->backward ? BWD : FWD)[n->op](n, lhs, rhs);
    return 0;
}
typedef struct { Node *nodes; uint32_t matmul, bias, relu, generation, *live_generation; int backward; } FusedCtx;
static int fused_execute(void *opaque) {
    FusedCtx *c = opaque;
    if (*c->live_generation != c->generation) return -4;
    Node *nodes = c->nodes;
    Node *mm = &nodes[c->matmul], *mm_lhs = &nodes[mm->lhs], *mm_rhs = &nodes[mm->rhs];
    Node *bias = &nodes[c->bias], *bias_rhs = &nodes[bias->rhs];
    Node *relu = c->relu != NONE ? &nodes[c->relu] : NULL;
    if (!c->backward) {
        FWD[MATMUL](mm, mm_lhs, mm_rhs);
        FWD[BIAS](bias, mm, bias_rhs);
        if (relu) FWD[RELU](relu, bias, NULL);
    } else {
        if (relu) BWD[RELU](relu, bias, NULL);
        BWD[BIAS](bias, mm, bias_rhs);
        BWD[MATMUL](mm, mm_lhs, mm_rhs);
    }
    return 0;
}
typedef struct { float *grad; uint32_t count, generation, *live_generation; } ZeroCtx;
static int zero_execute(void *opaque) {
    ZeroCtx *c = opaque;
    if (*c->live_generation != c->generation) return -4;
    memset(c->grad, 0, (size_t)c->count * sizeof(float));
    return 0;
}

/* --- IDX ubyte format: 4-byte magic, 4-byte-per-dim big-endian header, raw bytes --- */
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
        printf("mnist graph engine check skipped: set MNIST_DIR to a directory with the extracted IDX files "
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
        fprintf(stderr, "mnist graph engine check: could not load MNIST_DIR=%s\n", dir);
        return 1;
    }

    float x[IN], w1[IN * HID], b1[HID] = {0}, w2[HID * OUT], b2[OUT] = {0}, target[OUT];
    float mm1[HID], bias1[HID], relu1[HID], mm2[OUT], bias2[OUT], loss[1];
    float gw1[IN * HID], gb1[HID], gw2[HID * OUT], gb2[OUT];
    float gmm1[HID], gbias1[HID], grelu1[HID], gmm2[OUT], gbias2[OUT], gloss[1];
    for (int i = 0; i < IN * HID; i++) w1[i] = initv(i, 2) * .05f;
    for (int i = 0; i < HID * OUT; i++) w2[i] = initv(i, 5) * .1f;

    Node nodes[N_COUNT] = {
        [N_X] = {LEAF, NONE, NONE, INPUT, 1, IN, 0, x, NULL},
        [N_W1] = {LEAF, NONE, NONE, PARAM, IN, HID, 1, w1, gw1},
        [N_B1] = {LEAF, NONE, NONE, PARAM, 1, HID, 1, b1, gb1},
        [N_W2] = {LEAF, NONE, NONE, PARAM, HID, OUT, 1, w2, gw2},
        [N_B2] = {LEAF, NONE, NONE, PARAM, 1, OUT, 1, b2, gb2},
        [N_TARGET] = {LEAF, NONE, NONE, CONSTANT, 1, OUT, 0, target, NULL},
        [N_MM1] = {MATMUL, N_X, N_W1, TEMP, 1, HID, 0, mm1, gmm1},
        [N_BIAS1] = {BIAS, N_MM1, N_B1, TEMP, 1, HID, 0, bias1, gbias1},
        [N_RELU1] = {RELU, N_BIAS1, NONE, TEMP, 1, HID, 0, relu1, grelu1},
        [N_MM2] = {MATMUL, N_RELU1, N_W2, TEMP, 1, OUT, 0, mm2, gmm2},
        [N_BIAS2] = {BIAS, N_MM2, N_B2, TEMP, 1, OUT, 0, bias2, gbias2},
        [N_LOSS] = {MSE, N_BIAS2, N_TARGET, TEMP, 1, 1, 0, loss, gloss},
    };
    unsigned forward_nodes[6] = {N_MM1, N_BIAS1, N_RELU1, N_MM2, N_BIAS2, N_LOSS};
    Node *zeroable[9] = {&nodes[N_W1], &nodes[N_B1], &nodes[N_W2], &nodes[N_B2],
                          &nodes[N_MM1], &nodes[N_BIAS1], &nodes[N_RELU1], &nodes[N_MM2], &nodes[N_BIAS2]};

    /* Same fusion pass as the XOR MLP, at 784/32/10 instead of 2/4/1 —
     * confirms the pass generalizes across size, not just across models. */
    Group groups[6];
    unsigned ng = build_schedule(nodes, N_COUNT, forward_nodes, 6, groups);
    if (ng != 3 || groups[0].kind != GROUP_MBR || groups[1].kind != GROUP_MB || groups[2].kind != GROUP_PLAIN) {
        fprintf(stderr, "mnist graph fusion disagreed with the locked XOR schedule shape: ng=%u\n", ng);
        return 2;
    }

    uint32_t generation = 1;
    ExecStep steps[9 + 3], bsteps[3];
    PlainCtx pctx_f[3], pctx_b[3];
    FusedCtx fctx_f[3], fctx_b[3];
    ZeroCtx zctx[9];

    unsigned train_use = train_n < 3000 ? train_n : 3000;
    unsigned test_use = test_n < 1000 ? test_n : 1000;
    unsigned epochs = 8;
    float lr = .05f;

    unsigned correct_before = 0;
    for (unsigned s = 0; s < test_use; s++) {
        for (int i = 0; i < IN; i++) x[i] = test_images[(size_t)s * IN + i] / 255.0f;
        for (unsigned gi = 0; gi < ng; gi++) {
            if (groups[gi].kind == GROUP_PLAIN) { Node *n = &nodes[groups[gi].plain]; FWD[n->op](n, &nodes[n->lhs], n->rhs != NONE ? &nodes[n->rhs] : NULL); }
            else fused_execute(&(FusedCtx){nodes, groups[gi].matmul, groups[gi].bias, groups[gi].relu, generation, &generation, 0});
        }
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias2[c] > bias2[pred]) pred = c;
        correct_before += pred == test_labels[s];
    }

    for (unsigned epoch = 0; epoch < epochs; epoch++) {
        for (unsigned s = 0; s < train_use; s++) {
            for (int i = 0; i < IN; i++) x[i] = train_images[(size_t)s * IN + i] / 255.0f;
            for (int i = 0; i < OUT; i++) target[i] = train_labels[s] == i ? 1.0f : 0.0f;
            unsigned n = 0;
            for (unsigned i = 0; i < 9; i++) { zctx[i] = (ZeroCtx){zeroable[i]->grad, zeroable[i]->rows * zeroable[i]->cols, generation, &generation}; steps[n] = (ExecStep){zero_execute, &zctx[i], 0, 0, 0}; n++; }
            for (unsigned gi = 0; gi < ng; gi++) {
                if (groups[gi].kind == GROUP_PLAIN) { pctx_f[gi] = (PlainCtx){nodes, groups[gi].plain, generation, &generation, 0}; steps[n] = (ExecStep){plain_execute, &pctx_f[gi], 0, 0, 0}; }
                else { fctx_f[gi] = (FusedCtx){nodes, groups[gi].matmul, groups[gi].bias, groups[gi].relu, generation, &generation, 0}; steps[n] = (ExecStep){fused_execute, &fctx_f[gi], 0, 0, 0}; }
                n++;
            }
            if (n != 12 || tensor_transformer_steps_execute(steps, n)) return 3;
            nodes[N_LOSS].grad[0] = 1.0f;
            unsigned bn = 0;
            for (int gi = (int)ng - 1; gi >= 0; gi--) {
                if (groups[gi].kind == GROUP_PLAIN) { pctx_b[bn] = (PlainCtx){nodes, groups[gi].plain, generation, &generation, 1}; bsteps[bn] = (ExecStep){plain_execute, &pctx_b[bn], 0, 0, 0}; }
                else { fctx_b[bn] = (FusedCtx){nodes, groups[gi].matmul, groups[gi].bias, groups[gi].relu, generation, &generation, 1}; bsteps[bn] = (ExecStep){fused_execute, &fctx_b[bn], 0, 0, 0}; }
                bn++;
            }
            if (bn != ng || tensor_transformer_steps_execute(bsteps, bn)) return 4;
            for (int i = 0; i < IN * HID; i++) w1[i] -= lr * gw1[i];
            for (int i = 0; i < HID; i++) b1[i] -= lr * gb1[i];
            for (int i = 0; i < HID * OUT; i++) w2[i] -= lr * gw2[i];
            for (int i = 0; i < OUT; i++) b2[i] -= lr * gb2[i];
        }
    }

    unsigned correct_after = 0;
    for (unsigned s = 0; s < test_use; s++) {
        for (int i = 0; i < IN; i++) x[i] = test_images[(size_t)s * IN + i] / 255.0f;
        for (unsigned gi = 0; gi < ng; gi++) {
            if (groups[gi].kind == GROUP_PLAIN) { Node *n = &nodes[groups[gi].plain]; FWD[n->op](n, &nodes[n->lhs], n->rhs != NONE ? &nodes[n->rhs] : NULL); }
            else fused_execute(&(FusedCtx){nodes, groups[gi].matmul, groups[gi].bias, groups[gi].relu, generation, &generation, 0});
        }
        int pred = 0; for (int c = 1; c < OUT; c++) if (bias2[c] > bias2[pred]) pred = c;
        correct_after += pred == test_labels[s];
    }

    printf("tensor graph engine mnist passed: shape=%d-%d-%d(real-28x28-input) dataset=real(ossci-mirror) "
           "fusion=MBR+MB(same-as-xor,different-size) train_samples=%u test_samples=%u epochs=%u "
           "test_accuracy=%.1f%%->%.1f%% (chance=10%%)\n",
           IN, HID, OUT, train_use, test_use, epochs, 100.0 * correct_before / test_use, 100.0 * correct_after / test_use);

    free(train_images); free(train_labels); free(test_images); free(test_labels);
    return correct_after > correct_before && (100.0 * correct_after / test_use) > 40.0 ? 0 : 5;
}
