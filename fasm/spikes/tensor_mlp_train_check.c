#include "tensor_mlp_executor_spike.h"
#include <stdio.h>
#include <string.h>

/* Step 1: monolithic backward vs finite differences (same oracle pattern as
 * tensor_transformer_reference_spike.h's check()). */
static int fd_check(Block *b, float *param, float *grad, int count, int pos, const float *seed, const char *name) {
    float old = param[pos], eps = 1e-3f;
    mlp_forward(b);
    param[pos] = old + eps;
    mlp_forward(b);
    float plus = 0;
    for (int k = 0; k < OUT; k++) plus += b->out[k] * seed[k];
    param[pos] = old - eps;
    mlp_forward(b);
    float minus = 0;
    for (int k = 0; k < OUT; k++) minus += b->out[k] * seed[k];
    param[pos] = old;
    mlp_forward(b);
    float numeric = (plus - minus) / (2 * eps), analytic = grad[pos];
    float tol = 5e-3f * fmaxf(1, fmaxf(fabsf(numeric), fabsf(analytic)));
    if (fabsf(numeric - analytic) > tol) {
        fprintf(stderr, "mlp backward %s analytic=%g numeric=%g\n", name, analytic, numeric);
        return 1;
    }
    return 0;
}

static int same(const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) if (fabsf(a[i] - b[i]) > 1e-6f) return 0;
    return 1;
}

int main(void) {
    /* --- correctness: monolithic backward vs finite differences --- */
    Block b;
    mlp_initialize(&b);
    for (int i = 0; i < IN; i++) b.x[i] = mlp_initv(i, 1);
    float seed[OUT];
    for (int i = 0; i < OUT; i++) seed[i] = mlp_initv(i, 6);
    mlp_forward(&b);
    mlp_backward(&b, seed);
    if (fd_check(&b, b.x, b.dx, IN, 0, seed, "x") || fd_check(&b, b.w1, b.dw1, IN * HID, 3, seed, "w1") ||
        fd_check(&b, b.b1, b.db1, HID, 1, seed, "b1") || fd_check(&b, b.w2, b.dw2, HID * OUT, 2, seed, "w2") ||
        fd_check(&b, b.b2, b.db2, OUT, 0, seed, "b2"))
        return 1;

    /* --- correctness: scheduled executor backward vs monolithic reference --- */
    Block reference, actual;
    mlp_initialize(&reference);
    mlp_initialize(&actual);
    for (int i = 0; i < IN; i++) reference.x[i] = actual.x[i] = mlp_initv(i, 4);
    mlp_forward(&reference);
    mlp_backward(&reference, seed);
    mlp_forward(&actual);
    Scratch scratch = {0};
    ExecStep steps[8];
    Context ctx[8];
    uint32_t generation = 7;
    unsigned count = mlp_emit(&actual, &scratch, seed, &generation, steps, ctx);
    if (count != 8 || tensor_transformer_steps_execute(steps, count)) return 2;
    if (!same(reference.dx, actual.dx, IN) || !same(reference.dw1, actual.dw1, IN * HID) ||
        !same(reference.db1, actual.db1, HID) || !same(reference.dw2, actual.dw2, HID * OUT) ||
        !same(reference.db2, actual.db2, OUT))
        return 3;
    generation++;
    if (tensor_transformer_steps_execute(steps, count) != -4) return 4;

    /* --- training: same executor loop the transformer used, XOR truth table --- */
    static const float data[4][IN] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    static const float labels[4] = {0, 1, 1, 0};
    Block m;
    mlp_initialize(&m);
    Scratch train_scratch = {0};
    ExecStep train_steps[8];
    Context train_ctx[8];
    uint32_t train_generation = 100;
    float lr = .1f;
    for (unsigned epoch = 0; epoch < 4000; epoch++) {
        for (int p = 0; p < 4; p++) {
            memcpy(m.x, data[p], sizeof m.x);
            mlp_forward(&m);
            float error = m.out[0] - labels[p];
            float mlp_seed[OUT] = {error};
            unsigned c = mlp_emit(&m, &train_scratch, mlp_seed, &train_generation, train_steps, train_ctx);
            if (c != 8 || tensor_transformer_steps_execute(train_steps, c)) return 5;
            for (int i = 0; i < IN * HID; i++) m.w1[i] -= lr * m.dw1[i];
            for (int i = 0; i < HID; i++) m.b1[i] -= lr * m.db1[i];
            for (int i = 0; i < HID * OUT; i++) m.w2[i] -= lr * m.dw2[i];
            for (int i = 0; i < OUT; i++) m.b2[i] -= lr * m.db2[i];
        }
    }
    float predictions[4];
    int correct = 0;
    for (int p = 0; p < 4; p++) {
        memcpy(m.x, data[p], sizeof m.x);
        mlp_forward(&m);
        predictions[p] = m.out[0];
        correct += (predictions[p] >= .5f) == (labels[p] >= .5f);
    }
    printf("tensor mlp trained through transformer executor: shape=2-4-1 task=xor backward_actions=8 "
           "gradient_check=finite-difference+scheduled-exact epochs=4000 predictions=%.3f,%.3f,%.3f,%.3f correct=%d/4\n",
           predictions[0], predictions[1], predictions[2], predictions[3], correct);
    return correct == 4 ? 0 : 6;
}
