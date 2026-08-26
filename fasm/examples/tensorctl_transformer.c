#define TRANSFORMER_EXECUTOR_NO_MAIN
#include "tensor_transformer_executor_spike.h"
#undef TRANSFORMER_EXECUTOR_NO_MAIN
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "tensor_plan_trace_adapter_spike.h"

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

static void print_buffer_table(LiveBuffer *sched, unsigned n) {
    printf("  %-16s %6s %10s %8s\n", "buffer", "bytes", "live", "offset");
    for (unsigned i = 0; i < n; i++)
        printf("  %-16s %6u [%2u,%2u] %8u%s\n", sched[i].name, sched[i].bytes, sched[i].first, sched[i].last, sched[i].offset,
               sched[i].alias != NO_ALIAS ? " (view, no allocation)" : "");
}

/* Returns 1 if "save" was chosen, 0 if "rematerialize". Optionally prints
 * the full buffer table for whichever schedule was actually picked, so
 * "what got chosen" is a real, inspectable arena layout, not just a word. */
static int report_memory_plan(uint32_t budget, uint32_t counterfactual_budget, int verbose, PlanTrace *trace) {
    /* Not the same buffer count: "save" drops the separate scores_remat
     * entry entirely (scores lives across its full window instead), so
     * these must never share one length — that mismatch previously read
     * one element past the end of save_sched. */
    LiveBuffer remat_sched[] = {REAL_31_REMAT};
    LiveBuffer save_sched[] = {REAL_31_SAVE};
    unsigned remat_n = sizeof remat_sched / sizeof remat_sched[0];
    unsigned save_n = sizeof save_sched / sizeof save_sched[0];
    uint32_t remat_arena = arena_layout(remat_sched, remat_n);
    uint32_t save_arena = arena_layout(save_sched, save_n);
    uint32_t recompute_cost = 64; /* tiny 3x3x2x2 attention recompute, same estimate as the cost-planner spike */
    PlannerDecisionOutput decision = {"attention_scores", "budget overflow + recompute cost", "lowest combined planner cost",
        {{"save", save_arena, 0, 0, 1}, {"rematerialize", remat_arena, recompute_cost, 0, 1}}, 2};
    if (planner_choose_memory(&decision, budget, counterfactual_budget) || trace_from_planner(trace, &decision)) return 0;
    int save_chosen = decision.chosen == 0;
    printf("memory: budget=%u save_arena=%u remat_arena=%u recompute_cost=%u decision=%s(attention_scores)\n",
           budget, save_arena, remat_arena, recompute_cost, save_chosen ? "save" : "rematerialize");
    if (verbose) {
        printf("memory plan (chosen: %s, %u bytes, upper_bound=5760):\n", save_chosen ? "save" : "rematerialize", save_chosen ? save_arena : remat_arena);
        print_buffer_table(save_chosen ? save_sched : remat_sched, save_chosen ? save_n : remat_n);
    }
    return save_chosen;
}

static double now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec * 1e9 + ts.tv_nsec; }

/* Second real alternative for the planner to choose between: the usual
 * merge+CONTIGUOUS+matmul output projection vs the layout-aware head-wise
 * matmul that never materializes a contiguous copy at all. The decision
 * (see report_layout_decision below) is grounded in measured wall-clock,
 * not the byte-traffic count — 0 bytes copied vs 48 always favors
 * layout-aware on paper, but the measured segment is consistently ~15%
 * faster for standard at this real shape, so that's what gets chosen. And
 * unlike the first cut of this spike, the decision actually drives
 * execution: forward_layout()/the spliced 20-action backward below replace
 * forward()/emit()'s case7+case8 when the measurement favors layout-aware. */
static void forward_layout(Block *b) {
    mm(b->x, b->wq, b->qkv, T, M, QW);
    float scale = 1 / sqrtf(D);
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
        float mx = -INFINITY;
        for (int j = 0; j < T; j++) {
            float s = 0;
            for (int d = 0; d < D; d++) s += b->qkv[i * QW + h * D + d] * b->qkv[j * QW + M + h * D + d];
            s *= scale;
            b->score[(h * T + i) * T + j] = s;
            mx = fmaxf(mx, s);
        }
        float total = 0;
        for (int j = 0; j < T; j++) { float e = expf(b->score[(h * T + i) * T + j] - mx); b->prob[(h * T + i) * T + j] = e; total += e; }
        for (int j = 0; j < T; j++) b->prob[(h * T + i) * T + j] /= total;
        for (int d = 0; d < D; d++) {
            float s = 0;
            for (int j = 0; j < T; j++) s += b->prob[(h * T + i) * T + j] * b->qkv[j * QW + 2 * M + h * D + d];
            b->head[(h * T + i) * D + d] = s; /* no b->merge write */
        }
    }
    memset(b->proj, 0, sizeof b->proj);
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) for (int d = 0; d < D; d++) {
        float x = b->head[(h * T + i) * D + d];
        for (int j = 0; j < M; j++) b->proj[i * M + j] += x * b->wo[(h * D + d) * M + j];
    }
    for (int i = 0; i < T * M; i++) b->s1[i] = b->x[i] + b->proj[i];
    ln(b->s1, b->ln1, b->mean1, b->inv1);
    mm(b->ln1, b->w1, b->z1, T, M, F);
    for (int i = 0; i < T * F; i++) b->act[i] = b->z1[i] > 0 ? b->z1[i] : 0;
    mm(b->act, b->w2, b->ff, T, F, M);
    for (int i = 0; i < T * M; i++) b->s2[i] = b->ln1[i] + b->ff[i];
    ln(b->s2, b->out, b->mean2, b->inv2);
}

/* Splices the layout-aware dhead/dwo computation into the real scheduled
 * executor path in place of emit()'s case7 (mmback merge/wo -> dwo,dmerge)
 * and case8 (dmerge -> dhead split) — 21 actions become 20. This still runs
 * through tensor_transformer_steps_execute; nothing bypasses the executor,
 * emit() itself, or tensor_transformer_executor_spike.h — only two of its
 * steps are replaced with new ones after emit() builds them. */
typedef struct { Block *b; Scratch *s; uint32_t generation, *live_generation; } LayoutCtx;
static int layout_execute(void *opaque) {
    LayoutCtx *lc = opaque;
    if (*lc->live_generation != lc->generation) return -4;
    Block *b = lc->b;
    Scratch *s = lc->s;
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < T; i++) for (int d = 0; d < D; d++) {
            float g = 0;
            for (int j = 0; j < M; j++) g += s->dproj[i * M + j] * b->wo[(h * D + d) * M + j];
            s->dhead[(h * T + i) * D + d] = g;
        }
        for (int d = 0; d < D; d++) for (int j = 0; j < M; j++) {
            float g = 0;
            for (int i = 0; i < T; i++) g += b->head[(h * T + i) * D + d] * s->dproj[i * M + j];
            b->dwo[(h * D + d) * M + j] += g;
        }
    }
    return 0;
}
static unsigned emit_layout(Block *b, Scratch *s, const float *seed, uint32_t *generation, ExecStep *out_steps, Context *out_ctx, LayoutCtx *lctx) {
    ExecStep tmp_steps[21];
    Context tmp_ctx[21];
    unsigned n = emit(b, s, seed, generation, tmp_steps, tmp_ctx);
    if (n != 21) return 0;
    unsigned at = 0;
    for (unsigned i = 0; i < 16; i++) { out_ctx[at] = tmp_ctx[i]; out_steps[at] = tmp_steps[i]; out_steps[at].context = &out_ctx[at]; at++; }
    *lctx = (LayoutCtx){b, s, *generation, generation};
    out_steps[at] = (ExecStep){layout_execute, lctx, REVERSE_ACTION, 0, 0};
    at++;
    for (unsigned i = 18; i < 21; i++) { out_ctx[at] = tmp_ctx[i]; out_steps[at] = tmp_steps[i]; out_steps[at].context = &out_ctx[at]; at++; }
    return at; /* 20 */
}

/* same() (float array comparison) is already provided by
 * tensor_transformer_executor_spike.h, reused here as-is. */
static int cmp_double(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y ? 1 : 0; }

/* 1) numerical equivalence of the ACTUAL spliced executor path (not just
 * the monolithic math) against the standard 21-action path; 2) proof
 * forward_layout() never writes b->merge at all (poisoned, checked after);
 * 3) benchmark of the whole merge+output-projection segment specifically,
 * against real model state from an actual forward pass, not synthetic
 * isolated arrays or a single-shot timing (median-of-31, the same lesson
 * the earlier noise-dominated single-shot measurement already taught). */
/* A single call is far below clock_gettime's usable resolution (the
 * segment is a handful of nanoseconds of work) — loop internally and
 * return the per-call average, same reasoning as the size-sweep SIMD
 * profiler needing thousands of iterations per data point. */
enum { BENCH_INNER_ITERS = 500000 };
static Block bench_std_block, bench_lay_block;
static volatile float bench_sink; /* forces the compiler to keep the loop body: without this, .proj/.merge writes are
                                    * never observed after the function returns and can be dead-code-eliminated —
                                    * an earlier version of this benchmark reported 0.3ns for a real matmul, which
                                    * is exactly that: a proof the loop got optimized away, not a real measurement. */
static double bench_standard_segment(void) {
    double t0 = now_ns();
    for (int it = 0; it < BENCH_INNER_ITERS; it++) {
        for (int i = 0; i < T; i++) for (int h = 0; h < H; h++) for (int d = 0; d < D; d++) bench_std_block.merge[i * M + h * D + d] = bench_std_block.head[(h * T + i) * D + d];
        mm(bench_std_block.merge, bench_std_block.wo, bench_std_block.proj, T, M, M);
        bench_sink = bench_std_block.proj[0];
    }
    return (now_ns() - t0) / BENCH_INNER_ITERS;
}
static double bench_layout_segment(void) {
    double t0 = now_ns();
    for (int it = 0; it < BENCH_INNER_ITERS; it++) {
        memset(bench_lay_block.proj, 0, sizeof bench_lay_block.proj);
        for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) for (int d = 0; d < D; d++) {
            float x = bench_lay_block.head[(h * T + i) * D + d];
            for (int j = 0; j < M; j++) bench_lay_block.proj[i * M + j] += x * bench_lay_block.wo[(h * D + d) * M + j];
        }
        bench_sink = bench_lay_block.proj[0];
    }
    return (now_ns() - t0) / BENCH_INNER_ITERS;
}

/* --- profiling cache: an autotuning cache, not a cost model ---
 * The 62-trial live benchmark above is real but doesn't need re-running
 * every invocation. Cached per (shape, host arch, this-file's revision,
 * compiler identity, build config) — NOT per whole-repo hash, so an
 * unrelated README edit can't invalidate it, and NOT auto-expired by age,
 * since a three-day-old measurement isn't wrong just for being old. The
 * cache is advisory only: any read failure (missing, corrupt, unknown
 * format, key mismatch) falls back to a fresh measurement, never a crash. */
#define PROFILE_REVISION "v1" /* bump manually when forward_layout/layout_execute/bench_* change */
#ifndef TENSORCTL_BUILD_CONFIG
#define TENSORCTL_BUILD_CONFIG "unknown"
#endif

typedef struct { double median, min, max; unsigned n; } Stat;
static Stat measure_stat(double (*call)(void), int trials) {
    double t[64];
    if (trials > 64) trials = 64;
    for (int i = 0; i < trials; i++) t[i] = call();
    qsort(t, trials, sizeof(double), cmp_double);
    Stat s;
    s.median = t[trials / 2];
    s.min = t[0];
    s.max = t[trials - 1];
    s.n = (unsigned)trials;
    return s;
}
/* Overlapping [min,max] ranges (or a gap smaller than half the combined
 * spread) means the noise is at least as large as the effect — the planner
 * should say so instead of pretending it knows a winner. */
static int stats_uncertain(Stat a, Stat b) { return fabs(a.median - b.median) < (a.max - a.min + b.max - b.min) / 2.0; }

static void cache_dir_path(char *buf, size_t n) {
    const char *xdg = getenv("XDG_CACHE_HOME");
    const char *home = getenv("HOME");
    if (xdg && *xdg) snprintf(buf, n, "%s/tensorctl", xdg);
    else if (home && *home) snprintf(buf, n, "%s/Library/Caches/tensorctl", home); /* macOS platform default */
    else snprintf(buf, n, "/tmp/tensorctl-cache");
}
/* FNV-1a: not for anything security-sensitive, just to bucket distinct keys
 * into distinct files. Without this, a single fixed filename means writing
 * a second key (a different shape/host/build) overwrites the first key's
 * entry outright — every switch between two legitimate configurations
 * would thrash the cache instead of both coexisting. */
static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 16777619u; }
    return h;
}
static void cache_file_path(char *buf, size_t n, const char *key) {
    char dir[500];
    cache_dir_path(dir, sizeof dir);
    snprintf(buf, n, "%s/layout_decision.%08x.cache", dir, fnv1a(key));
}
static void build_cache_key(char *buf, size_t n) {
#if defined(__aarch64__)
    const char *host = "arm64";
#elif defined(__x86_64__)
    const char *host = "x86_64";
#else
    const char *host = "unknown";
#endif
    snprintf(buf, n, "shape=T%d-H%d-D%d-M%d-F%d;host=%s;revision=%s;compiler=%s;build=%s",
             T, H, D, M, F, host, PROFILE_REVISION, __VERSION__, TENSORCTL_BUILD_CONFIG);
}

typedef struct { char key[400]; Stat std_s, lay_s; long timestamp; } ProfileCache;

static void ensure_cache_dir(const char *dir) {
    char tmp[500];
    strncpy(tmp, dir, sizeof tmp - 1);
    tmp[sizeof tmp - 1] = 0;
    for (char *p = tmp + 1; *p; p++)
        if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
    mkdir(tmp, 0755);
}
/* Never fails loudly: any parse trouble or key mismatch is just a cache
 * miss, not an error — this file is advisory, not correctness-bearing. */
static int cache_read(const char *path, const char *expected_key, ProfileCache *out) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[600];
    int have_version = 0, key_ok = 0;
    memset(out, 0, sizeof *out);
    while (fgets(line, sizeof line, f)) {
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        char *val = eq + 1;
        size_t vlen = strlen(val);
        while (vlen && (val[vlen - 1] == '\n' || val[vlen - 1] == '\r')) val[--vlen] = 0;
        if (!strcmp(line, "format_version")) { if (strcmp(val, "1")) { fclose(f); return -1; } have_version = 1; }
        else if (!strcmp(line, "key")) { if (strcmp(val, expected_key)) { fclose(f); return -1; } key_ok = 1; strncpy(out->key, val, sizeof out->key - 1); }
        else if (!strcmp(line, "std_median")) out->std_s.median = atof(val);
        else if (!strcmp(line, "std_min")) out->std_s.min = atof(val);
        else if (!strcmp(line, "std_max")) out->std_s.max = atof(val);
        else if (!strcmp(line, "std_n")) out->std_s.n = (unsigned)atoi(val);
        else if (!strcmp(line, "lay_median")) out->lay_s.median = atof(val);
        else if (!strcmp(line, "lay_min")) out->lay_s.min = atof(val);
        else if (!strcmp(line, "lay_max")) out->lay_s.max = atof(val);
        else if (!strcmp(line, "lay_n")) out->lay_s.n = (unsigned)atoi(val);
        else if (!strcmp(line, "timestamp")) out->timestamp = atol(val);
    }
    fclose(f);
    if (!have_version || !key_ok || out->std_s.n == 0 || out->lay_s.n == 0) return -1;
    return 0;
}
static void cache_write_atomic(const char *path, const char *dir, const ProfileCache *r) {
    ensure_cache_dir(dir);
    char tmp_path[560];
    snprintf(tmp_path, sizeof tmp_path, "%s.tmp.%d", path, (int)getpid());
    FILE *f = fopen(tmp_path, "w");
    if (!f) return; /* advisory: if we can't write the cache, just skip it, don't fail the run */
    fprintf(f,
            "format_version=1\nkey=%s\n"
            "std_median=%.3f\nstd_min=%.3f\nstd_max=%.3f\nstd_n=%u\n"
            "lay_median=%.3f\nlay_min=%.3f\nlay_max=%.3f\nlay_n=%u\n"
            "timestamp=%ld\n",
            r->key, r->std_s.median, r->std_s.min, r->std_s.max, r->std_s.n,
            r->lay_s.median, r->lay_s.min, r->lay_s.max, r->lay_s.n, r->timestamp);
    fclose(f);
    if (rename(tmp_path, path) != 0) unlink(tmp_path);
}

static int report_layout_decision(int verbose, int reprofile, int no_cache, PlanTrace *trace) {
    uint32_t contiguous_copy_bytes = T * M * sizeof(float);

    /* (1) + (2): equivalence and no-copy proof, on the real spliced path.
     * Always run live, every time — cheap (a handful of forward/backward
     * calls, not 62 benchmark trials), and correctness is never cached. */
    Block std_b, lay_b;
    Scratch std_scratch = {0}, lay_scratch = {0};
    float seed[T * M];
    initialize(&std_b, seed);
    memcpy(&lay_b, &std_b, sizeof(Block));
    for (int i = 0; i < T * M; i++) lay_b.merge[i] = -999.0f; /* poison: forward_layout must never touch this */
    forward(&std_b);
    forward_layout(&lay_b);
    int merge_untouched = 1;
    for (int i = 0; i < T * M; i++) if (lay_b.merge[i] != -999.0f) merge_untouched = 0;
    int forward_matches = same(std_b.out, lay_b.out, T * M) && same(std_b.proj, lay_b.proj, T * M);

    uint32_t gen_std = 1, gen_lay = 1;
    ExecStep std_steps[21];
    Context std_ctx[21];
    unsigned std_count = emit(&std_b, &std_scratch, seed, &gen_std, std_steps, std_ctx);
    int std_ok = std_count == 21 && !tensor_transformer_steps_execute(std_steps, std_count);

    ExecStep lay_steps[20];
    Context lay_ctx[20];
    LayoutCtx lctx;
    unsigned lay_count = emit_layout(&lay_b, &lay_scratch, seed, &gen_lay, lay_steps, lay_ctx, &lctx);
    int lay_ok = lay_count == 20 && !tensor_transformer_steps_execute(lay_steps, lay_count);

    int backward_matches = std_ok && lay_ok && same(std_b.dx, lay_b.dx, T * M) && same(std_b.dwq, lay_b.dwq, M * QW) &&
                            same(std_b.dwo, lay_b.dwo, M * M) && same(std_b.dw1, lay_b.dw1, M * F) && same(std_b.dw2, lay_b.dw2, F * M);
    int verified = forward_matches && backward_matches && merge_untouched;

    /* (3): the expensive part — segment benchmark against real model state.
     * Try the cache first (unless disabled/forced); on any miss, measure
     * live and write back, still advisory (a write failure is silently
     * skipped, not an error). */
    char key[400];
    build_cache_key(key, sizeof key);
    char path[560], dir[500];
    cache_file_path(path, sizeof path, key);
    cache_dir_path(dir, sizeof dir);

    ProfileCache cached;
    int have_cache = !no_cache && !reprofile && cache_read(path, key, &cached) == 0;
    Stat std_s, lay_s;
    const char *source;
    long age_seconds = 0;
    if (have_cache) {
        std_s = cached.std_s;
        lay_s = cached.lay_s;
        source = "cached";
        age_seconds = (long)time(NULL) - cached.timestamp;
    } else {
        bench_std_block = std_b;
        bench_lay_block = lay_b;
        std_s = measure_stat(bench_standard_segment, 31);
        lay_s = measure_stat(bench_layout_segment, 31);
        source = "measured";
        if (!no_cache) {
            ProfileCache rec = {0};
            strncpy(rec.key, key, sizeof rec.key - 1);
            rec.std_s = std_s;
            rec.lay_s = lay_s;
            rec.timestamp = (long)time(NULL);
            cache_write_atomic(path, dir, &rec);
        }
    }

    /* The decision is grounded in the measured wall-clock, not the
     * byte-traffic count: 0 bytes copied vs 48 always favors layout-aware
     * on paper, but the actual measured segment (anti-DCE protected, real
     * model state) is consistently ~15-18% faster for standard at this real
     * T=3,H=2,D=2 shape — reproducible, not noise. When the gap IS within
     * noise, say so and fall back to the safer, longer-established path
     * (standard) instead of pretending a coin flip is a decision. */
    int uncertain = stats_uncertain(std_s, lay_s);
    MeasurementProvenance provenance = {0};
    snprintf(provenance.source, sizeof provenance.source, "%s", source);
#if defined(__aarch64__)
    snprintf(provenance.host, sizeof provenance.host, "arm64");
#elif defined(__x86_64__)
    snprintf(provenance.host, sizeof provenance.host, "x86_64");
#else
    snprintf(provenance.host, sizeof provenance.host, "unknown");
#endif
    snprintf(provenance.compiler, sizeof provenance.compiler, "%s", __VERSION__);
    snprintf(provenance.build, sizeof provenance.build, "%s", TENSORCTL_BUILD_CONFIG);
    snprintf(provenance.revision, sizeof provenance.revision, "%s", PROFILE_REVISION);
    provenance.samples = std_s.n < lay_s.n ? std_s.n : lay_s.n;
    provenance.median = std_s.median < lay_s.median ? std_s.median : lay_s.median;
    provenance.min = std_s.min < lay_s.min ? std_s.min : lay_s.min;
    provenance.max = std_s.max > lay_s.max ? std_s.max : lay_s.max;
    provenance.uncertain = uncertain;
    PlannerDecisionOutput decision = {"head_merge_layout", "verified and measurement confidence",
        uncertain ? "uncertain measurement falls back to standard" : "minimum measured median",
        {{"standard", 5760, (uint64_t)(std_s.median * 1000), 21, verified}, {"layout-aware", 5712, (uint64_t)(lay_s.median * 1000), 20, verified}}, 2,
        .provenance = provenance};
    planner_choose_measured(&decision, verified, uncertain);
    if (trace_from_planner(trace, &decision)) return 0;
    int layout_chosen = decision.chosen == 1;
    printf("layout: output projection — standard(merge+CONTIGUOUS+matmul) copies %u bytes/step, %s %.2fns/call [%.2f,%.2f] n=%u; "
           "layout-aware(head-wise matmul) copies 0 bytes/step, %s %.2fns/call [%.2f,%.2f] n=%u; "
           "decision=%s(%s, wall-clock not byte-traffic)\n",
           contiguous_copy_bytes, source, std_s.median, std_s.min, std_s.max, std_s.n,
           source, lay_s.median, lay_s.min, lay_s.max, lay_s.n,
           layout_chosen ? "layout-aware" : "standard", uncertain ? "uncertain" : "confident");
    if (!strcmp(source, "cached"))
        printf("  profile: source=cached age=%lds key=%s\n", age_seconds, key);
    else
        printf("  profile: source=measured key=%s%s\n", key, no_cache ? " (not cached: --no-profile-cache)" : "");
    printf("  verified: numerical_equivalence=%s merge_never_written=%s\n",
           (forward_matches && backward_matches) ? "yes" : "NO-MISMATCH", merge_untouched ? "yes" : "NO-COPY-DETECTED");
    if (verbose)
        printf("  the executor path above is the actual one used for backward when this decision picks layout-aware "
               "(21 actions -> 20: case7+case8 replaced by one layout_execute step)\n");
    return layout_chosen; /* verified==0 already forces this to 0 (falls back to standard) */
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

int run_transformer(int argc, char **argv) {
    unsigned epochs = 12000;
    uint32_t budget = 4928; /* today's real arena at the default policy */
    int plan_only = 0, reprofile = 0, no_cache = 0, explain = 0, counterfactual_set = 0;
    const char *export_path = NULL;
    uint32_t counterfactual_budget = budget;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--epochs=", 9)) epochs = (unsigned)atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--memory-budget=", 16)) budget = (uint32_t)atoi(argv[i] + 16);
        else if (!strcmp(argv[i], "--plan")) plan_only = 1;
        else if (!strcmp(argv[i], "--reprofile")) reprofile = 1;
        else if (!strcmp(argv[i], "--no-profile-cache")) no_cache = 1;
        else if (!strcmp(argv[i], "--explain-decisions") || !strcmp(argv[i], "--show-alternatives")) explain = 1;
        else if (!strncmp(argv[i], "--counterfactual-budget=", 24)) { counterfactual_budget = (uint32_t)atoi(argv[i] + 24); counterfactual_set = 1; }
        else if (!strncmp(argv[i], "--export=", 9)) { export_path = argv[i] + 9; plan_only = 1; }
        else if (!strcmp(argv[i], "--export") && i + 1 < argc) { export_path = argv[++i]; plan_only = 1; }
    }

    printf("model: transformer\n");
    printf("graph: shape=T3-M4-H2-D2-F6 forward=direct(not yet scheduled) "
           "backward=21-or-20 actions via shared executor tensor_transformer_steps_execute (depends on layout decision below) "
           "optimizer=4 updates(wq,wo,w1,w2, plain SGD, not yet scheduled)\n");
    if (!counterfactual_set) counterfactual_budget = budget;
    PlanTrace trace = {0};
    report_memory_plan(budget, counterfactual_budget, plan_only, &trace);
    int use_layout = report_layout_decision(plan_only, reprofile, no_cache, &trace);
    if (explain) trace_explain(stdout, &trace);
    if (export_path) { FILE *f = fopen(export_path, "wb"); if (!f) { fprintf(stderr, "tensorctl: cannot export plan trace to %s\n", export_path); return 1; } int failed = trace_export_tsv(f, &trace) || fclose(f); if (failed) { fprintf(stderr, "tensorctl: cannot export plan trace to %s\n", export_path); return 1; } }
    if (plan_only) return 0;

    Block b;
    Scratch scratch = {0};
    float seed[T * M], target[T * M];
    initialize(&b, seed);
    target_make(target);
    uint32_t generation = 41;
    ExecStep steps[21];
    Context ctx[21];
    LayoutCtx lctx;
    if (use_layout) forward_layout(&b); else forward(&b);
    float initial = mse_seed(&b, target, seed);

    double t0 = now_ns();
    float current = initial;
    for (unsigned epoch = 0; epoch < epochs; epoch++) {
        if (use_layout) forward_layout(&b); else forward(&b);
        current = mse_seed(&b, target, seed);
        unsigned count = use_layout ? emit_layout(&b, &scratch, seed, &generation, steps, ctx, &lctx) : emit(&b, &scratch, seed, &generation, steps, ctx);
        unsigned expected = use_layout ? 20 : 21;
        if (count != expected || trace.count < 2 || trace_verify_actions(&trace.decisions[1], count) || tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "tensorctl transformer: executor/trace consistency error\n"); return 1; }
        float lr = .02f;
        for (int i = 0; i < M * QW; i++) b.wq[i] -= lr * b.dwq[i];
        for (int i = 0; i < M * M; i++) b.wo[i] -= lr * b.dwo[i];
        for (int i = 0; i < M * F; i++) b.w1[i] -= lr * b.dw1[i];
        for (int i = 0; i < F * M; i++) b.w2[i] -= lr * b.dw2[i];
    }
    double t1 = now_ns();
    if (use_layout) forward_layout(&b); else forward(&b);
    current = mse_seed(&b, target, seed);

    printf("training: epochs=%u loss=%.6f->%.6f (executed via %s path, %u backward actions)\n",
           epochs, initial, current, use_layout ? "layout-aware" : "standard", use_layout ? 20u : 21u);
    printf("benchmark: total_ms=%.3f steps=%u ns_per_step=%.1f\n", (t1 - t0) / 1e6, epochs, (t1 - t0) / epochs);
    return current < initial * .25f ? 0 : 1;
}
