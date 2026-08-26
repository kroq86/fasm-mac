#include "tensor_mlp_executor_spike.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec * 1e9 + ts.tv_nsec; }

int run_mlp(int argc, char **argv) {
    unsigned epochs = 4000;
    float lr = .1f;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--epochs=", 9)) epochs = (unsigned)atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--lr=", 5)) lr = (float)atof(argv[i] + 5);
    }

    printf("model: mlp\n");
    printf("graph: shape=2-4-1 task=xor forward=direct(5 ops, not scheduled) "
           "backward=8 actions(zero=5 reverse=3) via shared executor tensor_transformer_steps_execute\n");
    printf("memory: not modeled for this shape yet (see tensorctl transformer for the memory-planner path)\n");

    static const float data[4][IN] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    static const float labels[4] = {0, 1, 1, 0};
    Block m;
    mlp_initialize(&m);
    Scratch scratch = {0};
    ExecStep steps[8];
    Context ctx[8];
    uint32_t generation = 100;

    double t0 = now_ns();
    for (unsigned epoch = 0; epoch < epochs; epoch++) {
        for (int p = 0; p < 4; p++) {
            memcpy(m.x, data[p], sizeof m.x);
            mlp_forward(&m);
            float error = m.out[0] - labels[p];
            float seed[OUT] = {error};
            unsigned c = mlp_emit(&m, &scratch, seed, &generation, steps, ctx);
            if (c != 8 || tensor_transformer_steps_execute(steps, c)) { fprintf(stderr, "tensorctl mlp: executor error\n"); return 1; }
            for (int i = 0; i < IN * HID; i++) m.w1[i] -= lr * m.dw1[i];
            for (int i = 0; i < HID; i++) m.b1[i] -= lr * m.db1[i];
            for (int i = 0; i < HID * OUT; i++) m.w2[i] -= lr * m.dw2[i];
            for (int i = 0; i < OUT; i++) m.b2[i] -= lr * m.db2[i];
        }
    }
    double t1 = now_ns();

    float predictions[4];
    int correct = 0;
    for (int p = 0; p < 4; p++) {
        memcpy(m.x, data[p], sizeof m.x);
        mlp_forward(&m);
        predictions[p] = m.out[0];
        correct += (predictions[p] >= .5f) == (labels[p] >= .5f);
    }
    printf("training: epochs=%u lr=%.3f predictions=%.3f,%.3f,%.3f,%.3f expected=0,1,1,0 correct=%d/4\n",
           epochs, lr, predictions[0], predictions[1], predictions[2], predictions[3], correct);
    printf("benchmark: total_ms=%.3f steps=%u ns_per_step=%.1f\n", (t1 - t0) / 1e6, epochs * 4, (t1 - t0) / (epochs * 4));
    return correct == 4 ? 0 : 1;
}
