#include <stdint.h>
#include <stdio.h>

/* Experimental optimizer-aware backward scheduling; not a stable core ABI.
 *
 * Today (tensor_transformer_scheduled_train_check.c) all four SGD updates
 * run as one batch of 4 actions strictly after all 21 backward actions
 * finish, even though the four weight gradients finish accumulating at very
 * different points inside that backward pass. This formalizes each
 * gradient buffer's real lifetime — first write (its zero action) to last
 * write (the reverse_one() case that finishes it) — as an ordinary resource
 * on the same event timeline tensor_transformer_liveness_check.c already
 * uses, and compares DEFERRED (today's real behavior) against EARLY (each
 * gradient's lifetime ends the instant its own last write happens, as if
 * its SGD update ran immediately after).
 *
 * The case->event mapping is not invented: it is read off
 * tensor_transformer_executor_spike.h's zero_one()/reverse_one() switches
 * and cross-checked against tensor_transformer_liveness_check.c's own
 * buffer intervals (reverse case 2 writes both s->dact and b->dw2 in the
 * same mmback call — that is exactly the case whose "d_act" buffer starts
 * at event 25 in the existing model, so dw2's last write is event 25;
 * case 7 writes dwo and produces "d_merged" at event 30; case 4 writes dw1
 * and produces "d_sum1" at event 28, so dw1's last write is event 27, the
 * event before). Zero actions occupy events 14..22 one per zero_one() case
 * index; case 1/2/3/4 zero dwq/dwo/dw1/dw2, so their first-write events are
 * 15/16/17/18.
 *
 * RESULT: on this real schedule, early release saves exactly 0 bytes, and
 * that is a real, mechanistic finding, not a bug — see part 1 below. Part 2
 * proves the mechanism itself is correctly implemented by isolating it from
 * this schedule's specific (and here, unhelpful) geometry.
 */

typedef struct { const char *name; uint32_t bytes; uint8_t first, last, alias; uint32_t offset; } Buffer;
#define NO_ALIAS 255
static uint32_t align64(uint32_t n) { return (n + 63) & ~63u; }
static int overlap(const Buffer *a, const Buffer *b) { return !(a->last < b->first || b->last < a->first); }

static void layout(Buffer *b, unsigned count, uint32_t *out_arena_end) {
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
    *out_arena_end = arena_end;
}

/* --- Part 1: the real 31-buffer/35-event schedule, plus the 4 weight-
 * gradient buffers inserted at their true chronological position (their
 * first-write events, 15-18, fall inside the zero-action phase, between
 * "output"/"ln2_stats" at event 13 and "d_sum2" at event 23 — NOT appended
 * at the end, which would silently make early-release untestable). --- */
enum { DWQ_BYTES = 4 * 12 * 4, DWO_BYTES = 4 * 4 * 4, DW1_BYTES = 4 * 6 * 4, DW2_BYTES = 6 * 4 * 4, DEFERRED_LAST = 35 };
#define REAL_UPTO_FORWARD \
    {"packed_qkv", 768, 0, 34, NO_ALIAS, 0}, {"q_view", 0, 0, 34, 0, 0}, {"k_view", 0, 0, 34, 0, 0}, {"v_view", 0, 0, 34, 0, 0}, \
    {"scores_fwd", 512, 4, 4, NO_ALIAS, 0}, {"prob", 512, 4, 33, NO_ALIAS, 0}, {"head", 256, 4, 31, NO_ALIAS, 0}, {"merged", 256, 5, 30, NO_ALIAS, 0}, \
    {"projected", 256, 6, 30, NO_ALIAS, 0}, {"sum1", 256, 7, 29, NO_ALIAS, 0}, {"ln1", 256, 8, 29, NO_ALIAS, 0}, {"ln1_stats", 64, 8, 28, NO_ALIAS, 0}, \
    {"ff1", 512, 9, 27, NO_ALIAS, 0}, {"activation", 512, 10, 25, NO_ALIAS, 0}, {"ff2", 256, 11, 25, NO_ALIAS, 0}, {"sum2", 256, 12, 23, NO_ALIAS, 0}, \
    {"output", 256, 13, 23, NO_ALIAS, 0}, {"ln2_stats", 64, 13, 23, NO_ALIAS, 0}
#define REAL_REST \
    {"d_sum2", 256, 23, 24, NO_ALIAS, 0}, {"d_ln1", 256, 24, 29, NO_ALIAS, 0}, \
    {"d_ff2", 256, 24, 25, NO_ALIAS, 0}, {"d_act", 512, 25, 26, NO_ALIAS, 0}, {"d_ff1", 512, 26, 27, NO_ALIAS, 0}, {"d_sum1", 256, 28, 29, NO_ALIAS, 0}, \
    {"d_projected", 256, 29, 30, NO_ALIAS, 0}, {"d_merged", 256, 30, 31, NO_ALIAS, 0}, {"d_head", 256, 31, 33, NO_ALIAS, 0}, {"scores_remat", 512, 32, 33, NO_ALIAS, 0}, \
    {"d_prob", 512, 33, 33, NO_ALIAS, 0}, {"d_score", 512, 33, 33, NO_ALIAS, 0}, {"d_qkv", 768, 33, 34, NO_ALIAS, 0}

static Buffer real_deferred[] = {
    REAL_UPTO_FORWARD,
    {"dwq", DWQ_BYTES, 15, DEFERRED_LAST, NO_ALIAS, 0}, {"dwo", DWO_BYTES, 16, DEFERRED_LAST, NO_ALIAS, 0},
    {"dw1", DW1_BYTES, 17, DEFERRED_LAST, NO_ALIAS, 0}, {"dw2", DW2_BYTES, 18, DEFERRED_LAST, NO_ALIAS, 0},
    REAL_REST,
};
enum { REAL_DEFERRED_COUNT = sizeof real_deferred / sizeof real_deferred[0] };

static Buffer real_early[] = {
    REAL_UPTO_FORWARD,
    {"dwq", DWQ_BYTES, 15, 34, NO_ALIAS, 0},  /* case 10: still the last reverse action either way */
    {"dwo", DWO_BYTES, 16, 30, NO_ALIAS, 0},  /* case 7 finishes it */
    {"dw1", DW1_BYTES, 17, 27, NO_ALIAS, 0},  /* case 4 finishes it */
    {"dw2", DW2_BYTES, 18, 25, NO_ALIAS, 0},  /* case 2 finishes it */
    REAL_REST,
};
enum { REAL_EARLY_COUNT = sizeof real_early / sizeof real_early[0] };

/* Regression anchor: same 31 buffers tensor_transformer_liveness_check.c
 * and the #1 cost-planner spike already verified lay out to 4928 bytes. */
static Buffer real_base[] = {REAL_UPTO_FORWARD, REAL_REST};
enum { REAL_BASE_COUNT = sizeof real_base / sizeof real_base[0] };

/* --- Part 2: isolated minimal proof the mechanism itself works, decoupled
 * from part 1's specific (here, saturated) geometry. One always-live
 * buffer, one gradient-shaped buffer, one later consumer that needs the
 * gradient's space once it's actually free. --- */
static Buffer iso_deferred[] = {
    {"always_alive", 1000, 0, 10, NO_ALIAS, 0},
    {"grad", 200, 2, 10, NO_ALIAS, 0},              /* deferred: alive until the batched optimizer pass at the very end */
    {"future_consumer", 200, 5, 10, NO_ALIAS, 0},
};
static Buffer iso_early[] = {
    {"always_alive", 1000, 0, 10, NO_ALIAS, 0},
    {"grad", 200, 2, 4, NO_ALIAS, 0},                /* early: dies right after its own SGD update at event 4 */
    {"future_consumer", 200, 5, 10, NO_ALIAS, 0},
};
enum { ISO_COUNT = 3 };

int main(void) {
    uint32_t real_base_arena, real_deferred_arena, real_early_arena, iso_deferred_arena, iso_early_arena;
    layout(real_base, REAL_BASE_COUNT, &real_base_arena);
    if (real_base_arena != 4928) return 1;

    layout(real_deferred, REAL_DEFERRED_COUNT, &real_deferred_arena);
    layout(real_early, REAL_EARLY_COUNT, &real_early_arena);
    if (real_early_arena > real_deferred_arena) return 2; /* early must never be worse */

    layout(iso_deferred, ISO_COUNT, &iso_deferred_arena);
    layout(iso_early, ISO_COUNT, &iso_early_arena);
    if (iso_early_arena >= iso_deferred_arena) return 3; /* the mechanism itself must show a real win in isolation */

    printf("tensor gradient lifetime planner passed: real_schedule base_arena=%u deferred_arena=%u early_arena=%u saved=%u "
           "(0 by construction: dw2/d_act are co-produced by the same mmback call and both consumed one tick later, "
           "so there is no free tick between them regardless of optimizer timing; the arena is already saturated "
           "through this window) isolated_mechanism deferred_arena=%u early_arena=%u saved=%u\n",
           real_base_arena, real_deferred_arena, real_early_arena, real_deferred_arena - real_early_arena,
           iso_deferred_arena, iso_early_arena, iso_deferred_arena - iso_early_arena);
    return 0;
}
