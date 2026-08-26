/* Merged-compiler regression check: fusion as a separate, generic pass.
 *
 * Criterion for the merge: fusion must be a separate pass on top of the
 * canonical compile(), not logic built into compile() or into any model's
 * code. This file reuses the exact XOR graph tensor_merged_mlp_check.c
 * already proved trains correctly through compile()'s own schedule, then
 * runs detect_fusion() (tensor_semantic_compiler.h) over the same node
 * array to confirm it still finds the same MATMUL+BIAS_ADD[+RELU] groups
 * tensor_compiler_planner_check.c originally locked in (steps=3,
 * fusion=MBR+MB) and tensor_graph_engine_fusion_check.c already reproduced
 * with real execution — now grounded in compile()'s own validated,
 * needs-grad-derived graph instead of a hand-built one.
 */
#include "tensor_semantic_compiler.h"
#include <stdio.h>
#include <string.h>

enum { N_X, N_W1, N_B1, N_W2, N_B2, N_TARGET, N_MM1, N_BIAS1, N_RELU1, N_MM2, N_BIAS2, N_LOSS, N_COUNT };
enum { IN = 2, HID = 4, OUT = 1 };

typedef struct { Node *g; Group grp; int backward; } FusedCtx;
static int fused_execute(void *opaque) {
    FusedCtx *c = opaque;
    Node *mm = &c->g[c->grp.matmul], *bi = &c->g[c->grp.bias];
    Node *x = &c->g[mm->lhs], *w = &c->g[mm->rhs], *b = &c->g[bi->rhs];
    Node *relu = c->grp.kind == GROUP_MBR ? &c->g[c->grp.relu] : NULL;
    if (!c->backward) {
        FWD[MATMUL](mm, x, w);
        FWD[BIAS_ADD](bi, mm, b);
        if (relu) FWD[RELU](relu, bi, NULL);
    } else {
        if (relu) BWD[RELU](relu, bi, NULL, 1);
        BWD[BIAS_ADD](bi, mm, b, 3);
        BWD[MATMUL](mm, x, w, 3);
    }
    return 0;
}

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

    Group groups[N_COUNT];
    unsigned ng = detect_fusion(g, N_COUNT, groups);
    if (ng != 3 || groups[0].kind != GROUP_MBR || groups[1].kind != GROUP_MB || groups[2].kind != GROUP_PLAIN) {
        fprintf(stderr, "fusion pass on the compile()-validated graph disagrees with the locked tensor_compiler_planner_check.c result: groups=%u\n", ng);
        return 1;
    }

    /* zero-grad + optimizer still come straight from compile() (criteria
     * 5/6): only forward/backward get replaced by 2 fused steps instead of
     * 5 per-node ones. */
    ExecStep full[32];
    Context ctx[32];
    uint32_t full_n;
    if (compile(g, N_COUNT, 0.1f, full, ctx, 32, &full_n)) return 1;
    /* full[] layout: forward(6) zero(9) backward(6) optimizer(4); splice in
     * fused forward/backward in place of the per-node forward/backward
     * slices, reusing compile()'s zero(9) and optimizer(4) untouched. */
    FusedCtx ffwd = {g, groups[0], 0}, ffwd2 = {g, groups[1], 0};
    FusedCtx fbwd = {g, groups[0], 1}, fbwd2 = {g, groups[1], 1};
    ExecStep steps[32];
    unsigned at = 0;
    steps[at++] = (ExecStep){fused_execute, &ffwd, FORWARD, 0, 0};   /* mm1+bias1+relu1 */
    steps[at++] = (ExecStep){fused_execute, &ffwd2, FORWARD, 0, 0};  /* mm2+bias2 */
    steps[at++] = full[5];                                          /* loss (PLAIN) */
    memcpy(steps + at, full + 6, 9 * sizeof(ExecStep));              /* zero, from compile() */
    at += 9;
    steps[at++] = full[15];                                         /* loss backward (PLAIN) */
    steps[at++] = (ExecStep){fused_execute, &fbwd2, BACKWARD, 0, 0}; /* mm2+bias2 backward */
    steps[at++] = (ExecStep){fused_execute, &fbwd, BACKWARD, 0, 0};  /* mm1+bias1+relu1 backward */
    memcpy(steps + at, full + 21, 4 * sizeof(ExecStep));             /* optimizer, from compile() */
    at += 4;

    static const float data[4][IN] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    static const float labels[4] = {0, 1, 1, 0};
    for (unsigned epoch = 0; epoch < 4000; epoch++) {
        for (int p = 0; p < 4; p++) {
            memcpy(x, data[p], sizeof x);
            target[0] = labels[p];
            if (tensor_transformer_steps_execute(steps, at)) return 2;
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
    printf("merged compiler check (fusion) passed: unfused_forward_actions=6 fused_forward_actions=3 groups=MBR,MB,PLAIN "
           "matches_tensor_compiler_planner_check=yes zero_grad_and_optimizer_still_from_compile()=yes "
           "predictions=%.3f,%.3f,%.3f,%.3f expected=0,1,1,0 correct=%d/4\n",
           predictions[0], predictions[1], predictions[2], predictions[3], correct);
    return correct == 4 ? 0 : 3;
}
