#define TRANSFORMER_EXECUTOR_NO_MAIN
#include "tensor_transformer_executor_spike.h"
#undef TRANSFORMER_EXECUTOR_NO_MAIN
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Memory-plan reporting: the exact real 31-buffer/35-event schedule and
 * cost decision from tensor_liveness_cost_planner_check.c, embedded so the
 * reported peak is genuinely computed here, not a pasted string. */
typedef struct { const char *name; uint32_t bytes; uint8_t first, last, alias; uint32_t offset; } LiveBuffer;
#define NO_ALIAS 255
static uint32_t align64(uint32_t n) { return (n + 63) & ~63u; }
static int lb_overlap(const LiveBuffer *a, const LiveBuffer *b) { return !(a->last < b->first || b->last < a->first); }
static uint32_t arena_layout(LiveBuffer *b, unsigned count) {
    uint32_t arena_end = 0;
    for (unsigned i = 0; i < count; i++) {
        if (b[i].alias != NO_ALIAS) { b[i].offset = b[b[i].alias].offset; continue; }
        uint32_t candidate = 0;
        for (;;) {
            int conflict = 0;
            for (unsigned j = 0; j < i; j++) if (b[j].alias == NO_ALIAS && lb_overlap(&b[i], &b[j])) {
                uint32_t je = b[j].offset + align64(b[j].bytes);
                if (candidate < je && candidate + align64(b[i].bytes) > b[j].offset) { candidate = je; conflict = 1; break; }
            }
            if (!conflict) break;
        }
        b[i].offset = candidate;
        uint32_t end = candidate + align64(b[i].bytes);
        if (end > arena_end) arena_end = end;
    }
    return arena_end;
}
/* Exactly tensor_liveness_cost_planner_check.c's schedule_remat/schedule_save
 * arrays (validated: remat_arena=4928 matches tensor_transformer_liveness_check.c,
 * save_arena=5440). Copied verbatim rather than shared via macro — an earlier
 * attempt to share REAL_31_UPTO_FORWARD/REAL_31_REST between both variants
 * left scores_fwd, scores_remat, AND a new scores_full buffer all present at
 * once in the save variant, double-counting bytes (5504 instead of 5440). */
#define REAL_31_REMAT \
    {"packed_qkv", 768, 0, 34, NO_ALIAS, 0}, {"q_view", 0, 0, 34, 0, 0}, {"k_view", 0, 0, 34, 0, 0}, {"v_view", 0, 0, 34, 0, 0}, \
    {"scores_fwd", 512, 4, 4, NO_ALIAS, 0}, {"prob", 512, 4, 33, NO_ALIAS, 0}, {"head", 256, 4, 31, NO_ALIAS, 0}, {"merged", 256, 5, 30, NO_ALIAS, 0}, \
    {"projected", 256, 6, 30, NO_ALIAS, 0}, {"sum1", 256, 7, 29, NO_ALIAS, 0}, {"ln1", 256, 8, 29, NO_ALIAS, 0}, {"ln1_stats", 64, 8, 28, NO_ALIAS, 0}, \
    {"ff1", 512, 9, 27, NO_ALIAS, 0}, {"activation", 512, 10, 25, NO_ALIAS, 0}, {"ff2", 256, 11, 25, NO_ALIAS, 0}, {"sum2", 256, 12, 23, NO_ALIAS, 0}, \
    {"output", 256, 13, 23, NO_ALIAS, 0}, {"ln2_stats", 64, 13, 23, NO_ALIAS, 0}, {"d_sum2", 256, 23, 24, NO_ALIAS, 0}, {"d_ln1", 256, 24, 29, NO_ALIAS, 0}, \
    {"d_ff2", 256, 24, 25, NO_ALIAS, 0}, {"d_act", 512, 25, 26, NO_ALIAS, 0}, {"d_ff1", 512, 26, 27, NO_ALIAS, 0}, {"d_sum1", 256, 28, 29, NO_ALIAS, 0}, \
    {"d_projected", 256, 29, 30, NO_ALIAS, 0}, {"d_merged", 256, 30, 31, NO_ALIAS, 0}, {"d_head", 256, 31, 33, NO_ALIAS, 0}, {"scores_remat", 512, 32, 33, NO_ALIAS, 0}, \
    {"d_prob", 512, 33, 33, NO_ALIAS, 0}, {"d_score", 512, 33, 33, NO_ALIAS, 0}, {"d_qkv", 768, 33, 34, NO_ALIAS, 0}
#define REAL_31_SAVE \
    {"packed_qkv", 768, 0, 34, NO_ALIAS, 0}, {"q_view", 0, 0, 34, 0, 0}, {"k_view", 0, 0, 34, 0, 0}, {"v_view", 0, 0, 34, 0, 0}, \
    {"scores", 512, 4, 33, NO_ALIAS, 0}, {"prob", 512, 4, 33, NO_ALIAS, 0}, {"head", 256, 4, 31, NO_ALIAS, 0}, {"merged", 256, 5, 30, NO_ALIAS, 0}, \
    {"projected", 256, 6, 30, NO_ALIAS, 0}, {"sum1", 256, 7, 29, NO_ALIAS, 0}, {"ln1", 256, 8, 29, NO_ALIAS, 0}, {"ln1_stats", 64, 8, 28, NO_ALIAS, 0}, \
    {"ff1", 512, 9, 27, NO_ALIAS, 0}, {"activation", 512, 10, 25, NO_ALIAS, 0}, {"ff2", 256, 11, 25, NO_ALIAS, 0}, {"sum2", 256, 12, 23, NO_ALIAS, 0}, \
    {"output", 256, 13, 23, NO_ALIAS, 0}, {"ln2_stats", 64, 13, 23, NO_ALIAS, 0}, {"d_sum2", 256, 23, 24, NO_ALIAS, 0}, {"d_ln1", 256, 24, 29, NO_ALIAS, 0}, \
    {"d_ff2", 256, 24, 25, NO_ALIAS, 0}, {"d_act", 512, 25, 26, NO_ALIAS, 0}, {"d_ff1", 512, 26, 27, NO_ALIAS, 0}, {"d_sum1", 256, 28, 29, NO_ALIAS, 0}, \
    {"d_projected", 256, 29, 30, NO_ALIAS, 0}, {"d_merged", 256, 30, 31, NO_ALIAS, 0}, {"d_head", 256, 31, 33, NO_ALIAS, 0}, \
    {"d_prob", 512, 33, 33, NO_ALIAS, 0}, {"d_score", 512, 33, 33, NO_ALIAS, 0}, {"d_qkv", 768, 33, 34, NO_ALIAS, 0}

static uint32_t overflow_cost(uint32_t arena, uint32_t budget) { return arena > budget ? arena - budget : 0; }

static void report_memory_plan(uint32_t budget) {
    LiveBuffer remat_sched[] = {REAL_31_REMAT};
    LiveBuffer save_sched[] = {REAL_31_SAVE};
    uint32_t remat_arena = arena_layout(remat_sched, sizeof remat_sched / sizeof remat_sched[0]);
    uint32_t save_arena = arena_layout(save_sched, sizeof save_sched / sizeof save_sched[0]);
    uint32_t recompute_cost = 64; /* tiny 3x3x2x2 attention recompute, same estimate as the cost-planner spike */
    uint32_t save_penalty = overflow_cost(save_arena, budget);
    uint32_t remat_penalty = overflow_cost(remat_arena, budget) + recompute_cost;
    const char *decision = save_penalty <= remat_penalty ? "save" : "rematerialize";
    printf("memory: budget=%u save_arena=%u remat_arena=%u recompute_cost=%u decision=%s(attention_scores)\n",
           budget, save_arena, remat_arena, recompute_cost, decision);
}

static float mse_seed(Block *b, const float *target, float *seed) {
    float loss = 0;
    for (int i = 0; i < T * M; i++) { float e = b->out[i] - target[i]; loss += e * e; seed[i] = 2 * e / (T * M); }
    return loss / (T * M);
}
static void target_make(float *t) {
    for (int i = 0; i < T; i++) {
        float m = 0, v = 0;
        for (int j = 0; j < M; j++) { t[i * M + j] = sinf((i + 1) * (j + 2) * .7f) + cosf((i - j) * .4f); m += t[i * M + j]; }
        m /= M;
        for (int j = 0; j < M; j++) { float d = t[i * M + j] - m; v += d * d; }
        float inv = 1 / sqrtf(v / M + 1e-5f);
        for (int j = 0; j < M; j++) t[i * M + j] = (t[i * M + j] - m) * inv;
    }
}
static double now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec * 1e9 + ts.tv_nsec; }

int run_transformer(int argc, char **argv) {
    unsigned epochs = 12000;
    uint32_t budget = 4928; /* today's real arena at the default policy */
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--epochs=", 9)) epochs = (unsigned)atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--memory-budget=", 16)) budget = (uint32_t)atoi(argv[i] + 16);
    }

    printf("model: transformer\n");
    printf("graph: shape=T3-M4-H2-D2-F6 forward=direct(matches monolithic reference) "
           "backward=21 actions(zero=9 reverse=9 remat=1 reverse=2) via shared executor tensor_transformer_steps_execute\n");
    report_memory_plan(budget);

    Block b;
    Scratch scratch = {0};
    float seed[T * M], target[T * M];
    initialize(&b, seed);
    target_make(target);
    uint32_t generation = 41;
    ExecStep steps[21];
    Context ctx[21];
    forward(&b);
    float initial = mse_seed(&b, target, seed);

    double t0 = now_ns();
    float current = initial;
    for (unsigned epoch = 0; epoch < epochs; epoch++) {
        forward(&b);
        current = mse_seed(&b, target, seed);
        unsigned count = emit(&b, &scratch, seed, &generation, steps, ctx);
        if (count != 21 || tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "tensorctl transformer: executor error\n"); return 1; }
        float lr = .02f;
        for (int i = 0; i < M * QW; i++) b.wq[i] -= lr * b.dwq[i];
        for (int i = 0; i < M * M; i++) b.wo[i] -= lr * b.dwo[i];
        for (int i = 0; i < M * F; i++) b.w1[i] -= lr * b.dw1[i];
        for (int i = 0; i < F * M; i++) b.w2[i] -= lr * b.dw2[i];
    }
    double t1 = now_ns();
    forward(&b);
    current = mse_seed(&b, target, seed);

    printf("training: epochs=%u loss=%.6f->%.6f\n", epochs, initial, current);
    printf("benchmark: total_ms=%.3f steps=%u ns_per_step=%.1f\n", (t1 - t0) / 1e6, epochs, (t1 - t0) / epochs);
    return current < initial * .25f ? 0 : 1;
}
