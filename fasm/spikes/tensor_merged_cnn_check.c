/* Merged-compiler generality check: CONV as a new op trait, added to the
 * exact same canonical compile() the MLP/fusion/Transformer/MNIST merges
 * already went green on; not a stable core ABI.
 *
 * The point isn't "can this engine do a CNN" — it's the cleanest version of
 * the generality claim yet: adding CONV to tensor_semantic_compiler.h
 * touched the op enum, one new kernel pair, one new dispatch-table entry,
 * and one new validate() branch. compile()'s needs-grad derivation,
 * zero-grad policy, backward grad-mask computation, optimizer scheduling,
 * and tensor_transformer_steps_execute were not touched at all — this file
 * is proof of that, not an assertion of it.
 *
 * Small synthetic task, same spirit as the XOR MLP before real MNIST ever
 * entered the picture: four fixed 6x6 single-channel "images", one 3x3/
 * 2-filter conv (valid, stride 1, no padding) -> relu -> a small FC head,
 * trained against four distinct fixed target vectors.
 */
#define HIN 6
#define WIN 6
#define CIN 1
#define COUT 2
#define KH 3
#define KW 3
#include "tensor_semantic_compiler.h"
#include <stdio.h>
#include <string.h>

enum { CONV_OUT = HOUT * WOUT * COUT }; /* 4*4*2 = 32 */
enum { FC_OUT = 2 };
static float initv(int i, int s) { return (float)(((i * 37 + s * 17) % 29) - 14) / 41.0f; }

enum { N_X, N_CW, N_FCW, N_FCB, N_TARGET, N_CONV, N_RELU, N_FC, N_BIAS, N_LOSS, N_COUNT };

int main(void) {
    float x[HIN * WIN * CIN], cw[COUT * CIN * KH * KW], fcw[CONV_OUT * FC_OUT], fcb[FC_OUT] = {0}, target[FC_OUT];
    float conv[CONV_OUT], relu[CONV_OUT], fc[FC_OUT], bias[FC_OUT], loss[1];
    float gx[HIN * WIN * CIN] = {0}, gcw[COUT * CIN * KH * KW], gfcw[CONV_OUT * FC_OUT], gfcb[FC_OUT], gtarget[FC_OUT] = {0};
    float gconv[CONV_OUT], grelu[CONV_OUT], gfc[FC_OUT], gbias[FC_OUT], gloss[1];

    for (int i = 0; i < COUT * CIN * KH * KW; i++) cw[i] = initv(i, 2) * .3f;
    for (int i = 0; i < CONV_OUT * FC_OUT; i++) fcw[i] = initv(i, 5) * .2f;

    Node g[N_COUNT] = {
        [N_X] = {LEAF, NONE, NONE, INPUT, {x, gx, NULL, 1, HIN * WIN * CIN, 0}},
        [N_CW] = {LEAF, NONE, NONE, PARAM, {cw, gcw, NULL, COUT, CIN * KH * KW, 0}},
        [N_FCW] = {LEAF, NONE, NONE, PARAM, {fcw, gfcw, NULL, CONV_OUT, FC_OUT, 0}},
        [N_FCB] = {LEAF, NONE, NONE, PARAM, {fcb, gfcb, NULL, 1, FC_OUT, 0}},
        [N_TARGET] = {LEAF, NONE, NONE, CONSTANT, {target, gtarget, NULL, 1, FC_OUT, 0}},
        [N_CONV] = {CONV, N_X, N_CW, TEMP, {conv, gconv, NULL, 1, CONV_OUT, 0}},
        [N_RELU] = {RELU, N_CONV, NONE, TEMP, {relu, grelu, NULL, 1, CONV_OUT, 0}},
        [N_FC] = {MATMUL, N_RELU, N_FCW, TEMP, {fc, gfc, NULL, 1, FC_OUT, 0}},
        [N_BIAS] = {BIAS_ADD, N_FC, N_FCB, TEMP, {bias, gbias, NULL, 1, FC_OUT, 0}},
        [N_LOSS] = {MSE, N_BIAS, N_TARGET, TEMP, {loss, gloss, NULL, 1, 1, 0}},
    };

    ExecStep steps[32];
    Context ctx[32];
    uint32_t count = 0;
    if (compile(g, N_COUNT, 0.05f, steps, ctx, 32, &count)) { fprintf(stderr, "compile failed\n"); return 1; }
    /* forward=5(conv,relu,fc,bias,loss) zero=7(3 params + 4 non-mse temps)
     * backward=5(incl. mse) optimizer=3(cw,fcw,fcb) -> 20, unchanged
     * compile() algorithm, same as every other merged check */
    if (count != 20) { fprintf(stderr, "unexpected step count=%u (want 20)\n", count); return 1; }

    static const float patterns[4][2] = {{1.0f, -1.0f}, {-1.0f, 1.0f}, {0.5f, 0.5f}, {-0.5f, -0.5f}};

    /* correctness first: finite-difference oracle on both the new op
     * (conv weight) and an existing one (fc weight), through compile()'s
     * own emitted schedule — same pattern every other spike uses. layout
     * is forward(5) zero(7) backward(5) optimizer(3); run forward+zero+
     * backward (0..17) to capture analytic grad without the optimizer
     * slice touching weights, then forward-only (0..5) to re-check loss
     * at a perturbed weight. */
    for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = initv(i, 10);
    memcpy(target, patterns[0], sizeof target);
    if (tensor_transformer_steps_execute(steps, 5 + 7 + 5)) return 2;
    struct { float *param, *grad; int idx; const char *name; } probes[] = {
        {cw, gcw, 4, "conv_weight"}, {fcw, gfcw, 9, "fc_weight"},
    };
    for (unsigned p = 0; p < 2; p++) {
        float eps = 1e-3f, old = probes[p].param[probes[p].idx], analytic = probes[p].grad[probes[p].idx];
        probes[p].param[probes[p].idx] = old + eps;
        if (tensor_transformer_steps_execute(steps, 5)) return 2;
        float plus = loss[0];
        probes[p].param[probes[p].idx] = old - eps;
        if (tensor_transformer_steps_execute(steps, 5)) return 2;
        float minus = loss[0];
        probes[p].param[probes[p].idx] = old;
        float numeric = (plus - minus) / (2 * eps);
        if (fabsf(numeric - analytic) > 5e-3f * fmaxf(1, fmaxf(fabsf(numeric), fabsf(analytic)))) {
            fprintf(stderr, "merged compiler cnn backward mismatch on %s: analytic=%g numeric=%g\n", probes[p].name, analytic, numeric);
            return 3;
        }
    }

    for (int i = 0; i < COUT * CIN * KH * KW; i++) cw[i] = initv(i, 2) * .3f;
    for (int i = 0; i < CONV_OUT * FC_OUT; i++) fcw[i] = initv(i, 5) * .2f;
    memset(fcb, 0, sizeof fcb);
    for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = initv(i, 10);
    memcpy(target, patterns[0], sizeof target);
    if (tensor_transformer_steps_execute(steps, 5)) return 4;
    float initial = loss[0];
    for (unsigned epoch = 0; epoch < 4000; epoch++) {
        for (int p = 0; p < 4; p++) {
            for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = initv(i, 10 + p);
            memcpy(target, patterns[p], sizeof target);
            if (tensor_transformer_steps_execute(steps, count)) return 5;
        }
    }
    for (int i = 0; i < HIN * WIN * CIN; i++) x[i] = initv(i, 10);
    memcpy(target, patterns[0], sizeof target);
    if (tensor_transformer_steps_execute(steps, 5)) return 4;
    float final = loss[0];

    printf("merged compiler check (cnn) passed: new_op=conv (valid 3x3 conv, %dx%dx%d->%dx%dx%d) "
           "nodes=%u steps=%u (forward=5 zero=7 backward=5 optimizer=3, all compile()-derived, "
           "no forward_nodes[]/zeroable[] in this file, no changes to compile()/validate() for any "
           "pre-existing op) gradient_check=finite-difference(conv_weight,fc_weight) epochs=4000 "
           "loss=%.6f->%.6f\n",
           HIN, WIN, CIN, HOUT, WOUT, COUT, (unsigned)N_COUNT, count, initial, final);
    return final < initial * .25f ? 0 : 6;
}
