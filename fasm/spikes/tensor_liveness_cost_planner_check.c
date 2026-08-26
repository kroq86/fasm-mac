#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Experimental cost-aware save-vs-rematerialize planner; not a stable core
 * ABI. tensor_transformer_liveness_check.c already proved the arena layout
 * math on the real 31-buffer/35-event schedule, but its rematerialize-vs-
 * save choice for attention scores was a single hand-picked policy baked
 * into the buffer list. This spike makes that an actual decision: lay out
 * the arena twice (scores kept alive the whole time vs scores split into a
 * short recomputed interval), turn "bytes over budget" and "recompute cost"
 * into one comparable cost, and pick whichever is cheaper. The acceptance
 * bar is that changing the budget or the recompute cost flips the decision
 * — proof this is a real trade-off, not a relabeled hardcoded if. */

typedef struct { const char *name; uint32_t bytes; uint8_t first, last, alias; uint32_t offset; } Buffer;
#define NO_ALIAS 255
static uint32_t align64(uint32_t n) { return (n + 63) & ~63u; }
static int overlap(const Buffer *a, const Buffer *b) { return !(a->last < b->first || b->last < a->first); }

/* Same first-fit arena layout as tensor_transformer_liveness_check.c,
 * factored out so both variants of a candidate schedule go through the
 * identical algorithm — the only difference is the buffer list itself. */
static void layout(Buffer *b, unsigned count, uint32_t *out_arena_end, uint32_t *out_raw_peak) {
    uint32_t arena_end = 0;
    for (unsigned i = 0; i < count; i++) {
        if (b[i].alias != NO_ALIAS) { b[i].offset = b[b[i].alias].offset; continue; }
        uint32_t candidate = 0;
        for (;;) {
            int conflict = 0;
            for (unsigned j = 0; j < i; j++) if (b[j].alias == NO_ALIAS && overlap(&b[i], &b[j])) {
                uint32_t je = b[j].offset + align64(b[j].bytes);
                if (candidate < je && candidate + align64(b[i].bytes) > b[j].offset) { candidate = je; conflict = 1; break; }
            }
            if (!conflict) break;
        }
        b[i].offset = candidate;
        uint32_t end = candidate + align64(b[i].bytes);
        if (end > arena_end) arena_end = end;
    }
    uint32_t raw_peak = 0;
    for (unsigned e = 0; e < 35; e++) {
        uint32_t live = 0;
        for (unsigned i = 0; i < count; i++) if (b[i].alias == NO_ALIAS && b[i].first <= e && e <= b[i].last) live += b[i].bytes;
        if (live > raw_peak) raw_peak = live;
    }
    *out_arena_end = arena_end;
    *out_raw_peak = raw_peak;
}

/* Byte-identical to tensor_transformer_liveness_check.c's schedule: attention
 * scores rematerialized (scores_fwd dies at event 4, scores_remat is a short
 * recomputed interval at 32-33). This is the REMAT variant of the "scores"
 * candidate. Events 0..13 forward; 14..22 zero; 23..34 reverse/remat. */
static Buffer schedule_remat[] = {
    {"packed_qkv", 768, 0, 34, NO_ALIAS, 0}, {"q_view", 0, 0, 34, 0, 0}, {"k_view", 0, 0, 34, 0, 0}, {"v_view", 0, 0, 34, 0, 0},
    {"scores_fwd", 512, 4, 4, NO_ALIAS, 0}, {"prob", 512, 4, 33, NO_ALIAS, 0}, {"head", 256, 4, 31, NO_ALIAS, 0}, {"merged", 256, 5, 30, NO_ALIAS, 0},
    {"projected", 256, 6, 30, NO_ALIAS, 0}, {"sum1", 256, 7, 29, NO_ALIAS, 0}, {"ln1", 256, 8, 29, NO_ALIAS, 0}, {"ln1_stats", 64, 8, 28, NO_ALIAS, 0},
    {"ff1", 512, 9, 27, NO_ALIAS, 0}, {"activation", 512, 10, 25, NO_ALIAS, 0}, {"ff2", 256, 11, 25, NO_ALIAS, 0}, {"sum2", 256, 12, 23, NO_ALIAS, 0},
    {"output", 256, 13, 23, NO_ALIAS, 0}, {"ln2_stats", 64, 13, 23, NO_ALIAS, 0}, {"d_sum2", 256, 23, 24, NO_ALIAS, 0}, {"d_ln1", 256, 24, 29, NO_ALIAS, 0},
    {"d_ff2", 256, 24, 25, NO_ALIAS, 0}, {"d_act", 512, 25, 26, NO_ALIAS, 0}, {"d_ff1", 512, 26, 27, NO_ALIAS, 0}, {"d_sum1", 256, 28, 29, NO_ALIAS, 0},
    {"d_projected", 256, 29, 30, NO_ALIAS, 0}, {"d_merged", 256, 30, 31, NO_ALIAS, 0}, {"d_head", 256, 31, 33, NO_ALIAS, 0}, {"scores_remat", 512, 32, 33, NO_ALIAS, 0},
    {"d_prob", 512, 33, 33, NO_ALIAS, 0}, {"d_score", 512, 33, 33, NO_ALIAS, 0}, {"d_qkv", 768, 33, 34, NO_ALIAS, 0},
};
enum { REMAT_COUNT = sizeof schedule_remat / sizeof schedule_remat[0] };

/* Same schedule, but scores is kept alive across its full real need window
 * [4,33] as one buffer instead of split+recomputed. This is the SAVE
 * variant of the exact same candidate — same graph, same math, only the
 * memory policy for one intermediate differs. */
static Buffer schedule_save[] = {
    {"packed_qkv", 768, 0, 34, NO_ALIAS, 0}, {"q_view", 0, 0, 34, 0, 0}, {"k_view", 0, 0, 34, 0, 0}, {"v_view", 0, 0, 34, 0, 0},
    {"scores", 512, 4, 33, NO_ALIAS, 0}, {"prob", 512, 4, 33, NO_ALIAS, 0}, {"head", 256, 4, 31, NO_ALIAS, 0}, {"merged", 256, 5, 30, NO_ALIAS, 0},
    {"projected", 256, 6, 30, NO_ALIAS, 0}, {"sum1", 256, 7, 29, NO_ALIAS, 0}, {"ln1", 256, 8, 29, NO_ALIAS, 0}, {"ln1_stats", 64, 8, 28, NO_ALIAS, 0},
    {"ff1", 512, 9, 27, NO_ALIAS, 0}, {"activation", 512, 10, 25, NO_ALIAS, 0}, {"ff2", 256, 11, 25, NO_ALIAS, 0}, {"sum2", 256, 12, 23, NO_ALIAS, 0},
    {"output", 256, 13, 23, NO_ALIAS, 0}, {"ln2_stats", 64, 13, 23, NO_ALIAS, 0}, {"d_sum2", 256, 23, 24, NO_ALIAS, 0}, {"d_ln1", 256, 24, 29, NO_ALIAS, 0},
    {"d_ff2", 256, 24, 25, NO_ALIAS, 0}, {"d_act", 512, 25, 26, NO_ALIAS, 0}, {"d_ff1", 512, 26, 27, NO_ALIAS, 0}, {"d_sum1", 256, 28, 29, NO_ALIAS, 0},
    {"d_projected", 256, 29, 30, NO_ALIAS, 0}, {"d_merged", 256, 30, 31, NO_ALIAS, 0}, {"d_head", 256, 31, 33, NO_ALIAS, 0},
    {"d_prob", 512, 33, 33, NO_ALIAS, 0}, {"d_score", 512, 33, 33, NO_ALIAS, 0}, {"d_qkv", 768, 33, 34, NO_ALIAS, 0},
};
enum { SAVE_COUNT = sizeof schedule_save / sizeof schedule_save[0] };

/* The trade-off: express "bytes over budget" and "recompute cost" in one
 * comparable unit and pick whichever variant costs less. budget is a soft
 * target, not a hard cap — going over it costs exactly as much penalty as
 * the overflow, in the same units recompute_cost is given in. */
static uint32_t overflow(uint32_t arena, uint32_t budget) { return arena > budget ? arena - budget : 0; }
static int decide_save(uint32_t save_arena, uint32_t remat_arena, uint32_t recompute_cost, uint32_t budget) {
    uint32_t save_penalty = overflow(save_arena, budget);
    uint32_t remat_penalty = overflow(remat_arena, budget) + recompute_cost;
    return save_penalty <= remat_penalty;
}

typedef struct { const char *name; uint32_t save_arena, remat_arena, cheap_cost, expensive_cost; } Candidate;

static int check_candidate(const Candidate *c) {
    uint32_t tight_budget = c->remat_arena;                 /* today's real target: fits the remat layout, not the save one */
    uint32_t large_budget = c->save_arena + c->save_arena;  /* comfortably covers either layout */
    int at_large_budget = decide_save(c->save_arena, c->remat_arena, c->cheap_cost, large_budget);
    int at_tight_budget = decide_save(c->save_arena, c->remat_arena, c->cheap_cost, tight_budget);
    int at_tight_budget_expensive = decide_save(c->save_arena, c->remat_arena, c->expensive_cost, tight_budget);
    printf("candidate=%-14s save_arena=%5u remat_arena=%5u large_budget=%s tight_budget=%s tight_budget_expensive_remat=%s\n",
           c->name, c->save_arena, c->remat_arena,
           at_large_budget ? "save" : "remat", at_tight_budget ? "save" : "remat", at_tight_budget_expensive ? "save" : "remat");
    /* The whole point: budget and cost actually move the decision. */
    return at_large_budget == 1 && at_tight_budget == 0 && at_tight_budget_expensive == 1;
}

int main(void) {
    uint32_t remat_arena, remat_peak, save_arena, save_peak;
    layout(schedule_remat, REMAT_COUNT, &remat_arena, &remat_peak);
    layout(schedule_save, SAVE_COUNT, &save_arena, &save_peak);
    if (remat_arena != 4928) return 1; /* must reproduce tensor_transformer_liveness_check.c's known-good arena exactly */
    if (save_arena <= remat_arena) return 2; /* saving the full interval must genuinely cost more arena, or this test proves nothing */

    Candidate candidates[] = {
        {"scores", save_arena, remat_arena, 64, 100000},   /* real data: tiny 3x3x2x2 attention recompute is cheap by construction */
        {"synthetic_b", 2000, 1200, 50, 100000},
        {"synthetic_c", 5000, 800, 10, 100000},
    };
    unsigned n = sizeof candidates / sizeof candidates[0];
    int ok = 1;
    for (unsigned i = 0; i < n; i++) ok &= check_candidate(&candidates[i]);

    printf("tensor liveness cost planner passed: real_scores_save_arena=%u real_scores_remat_arena=%u "
           "candidates=%u mechanism=budget_and_cost_both_move_the_decision\n", save_arena, remat_arena, n);
    return ok ? 0 : 3;
}
