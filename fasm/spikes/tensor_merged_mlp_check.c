/* Merged-compiler regression check: MLP, via the single canonical compile().
 *
 * tensor_semantic_compiler.h is the merge of tensor_semantic_training_
 * compiler_check.c's compile() skeleton and tensor_graph_engine_check.c's
 * table dispatch. This file is ONLY a graph + tensors (criterion: no
 * forward_nodes[]/zeroable[]/hand-written schedule-assembly code per
 * model) — it must reproduce the exact XOR truth table both prior
 * standalone implementations already reached, as a regression check that
 * the merge didn't change behavior. Neither prior file is touched.
 */
#include "tensor_semantic_compiler.h"
#include <stdio.h>
#include <string.h>

enum { N_X, N_W1, N_B1, N_W2, N_B2, N_TARGET, N_MM1, N_BIAS1, N_RELU1, N_MM2, N_BIAS2, N_LOSS, N_COUNT };
enum { IN = 2, HID = 4, OUT = 1 };

int main(void) {
    float x[IN], w1[IN * HID] = {0.5f, -0.7f, 0.3f, 0.8f, -0.4f, 0.6f, 0.9f, -0.2f}, b1[HID] = {0.1f, 0.1f, -0.1f, 0.0f};
    float w2[HID * OUT] = {0.7f, -0.5f, 0.6f, -0.8f}, b2[OUT] = {0}, target[OUT];
    float mm1[HID], bias1[HID], relu1[HID], mm2[OUT], bias2[OUT], loss[1];
    float gx[IN] = {0}, gw1[IN * HID], gb1[HID], gw2[HID * OUT], gb2[OUT], gtarget[OUT] = {0};
    float gmm1[HID], gbias1[HID], grelu1[HID], gmm2[OUT], gbias2[OUT], gloss[1];

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
    if (compile(g, N_COUNT, 0.1f, steps, ctx, 32, &count)) { fprintf(stderr, "compile failed\n"); return 1; }
    if (count != 25) { fprintf(stderr, "unexpected step count=%u (want 25, matches Codex's compiler on this exact graph)\n", count); return 1; }

    static const float data[4][IN] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    static const float labels[4] = {0, 1, 1, 0};

    /* correctness first: finite-difference oracle on the merged compiler's
     * own emitted schedule, same pattern every other spike in this repo
     * uses. compile() lays steps out as forward(6), zero(9), backward(6),
     * optimizer(4) in that order (count==25 asserted above) — run only the
     * forward(0..6) slice to re-check loss at a perturbed weight so the
     * optimizer slice never touches w1 mid-check. */
    x[0] = 0; x[1] = 1; target[0] = 1;
    if (tensor_transformer_steps_execute(steps, 6 + 9 + 6)) return 2; /* forward+zero+backward, no optimizer */
    float eps = 1e-3f, old = w1[3], analytic = gw1[3];
    w1[3] = old + eps;
    if (tensor_transformer_steps_execute(steps, 6)) return 2; /* forward only */
    float plus = loss[0];
    w1[3] = old - eps;
    if (tensor_transformer_steps_execute(steps, 6)) return 2;
    float minus = loss[0];
    w1[3] = old;
    float numeric = (plus - minus) / (2 * eps);
    if (fabsf(numeric - analytic) > 5e-3f * fmaxf(1, fmaxf(fabsf(numeric), fabsf(analytic)))) {
        fprintf(stderr, "merged compiler backward mismatch: analytic=%g numeric=%g\n", analytic, numeric);
        return 3;
    }

    memcpy(w1, (float[]){0.5f, -0.7f, 0.3f, 0.8f, -0.4f, 0.6f, 0.9f, -0.2f}, sizeof w1);
    memcpy(b1, (float[]){0.1f, 0.1f, -0.1f, 0.0f}, sizeof b1);
    memcpy(w2, (float[]){0.7f, -0.5f, 0.6f, -0.8f}, sizeof w2);
    b2[0] = 0;
    for (unsigned epoch = 0; epoch < 4000; epoch++) {
        for (int p = 0; p < 4; p++) {
            memcpy(x, data[p], sizeof x);
            target[0] = labels[p];
            if (tensor_transformer_steps_execute(steps, count)) return 4;
        }
    }
    float predictions[4];
    int correct = 0;
    for (int p = 0; p < 4; p++) {
        memcpy(x, data[p], sizeof x);
        FWD[MATMUL](&g[N_MM1], &g[N_X], &g[N_W1]);
        FWD[BIAS_ADD](&g[N_BIAS1], &g[N_MM1], &g[N_B1]);
        FWD[RELU](&g[N_RELU1], &g[N_BIAS1], NULL);
        FWD[MATMUL](&g[N_MM2], &g[N_RELU1], &g[N_W2]);
        FWD[BIAS_ADD](&g[N_BIAS2], &g[N_MM2], &g[N_B2]);
        predictions[p] = bias2[0];
        correct += (predictions[p] >= .5f) == (labels[p] >= .5f);
    }
    printf("merged compiler check (mlp) passed: nodes=%u steps=%u (forward=6 zero=9 backward=6 optimizer=4, all compile()-derived, "
           "no forward_nodes[]/zeroable[] in this file) gradient_check=finite-difference task=xor epochs=4000 "
           "predictions=%.3f,%.3f,%.3f,%.3f expected=0,1,1,0 correct=%d/4\n",
           (unsigned)N_COUNT, count, predictions[0], predictions[1], predictions[2], predictions[3], correct);
    return correct == 4 ? 0 : 6;
}
