/* Native-path fixed/marginal cost decomposition, analogous to
 * phase1_kv_cache_minigunpoint_32token_amortized_latency.py's PyTorch-
 * reference-environment measurement, but through the real FASM/Accelerate
 * tensor-kv-handoff binary instead. Same field names (fixed_cost_sender_
 * plus_adapter, marginal_cost_cache_query) for direct comparison.
 *
 * Model/tokenizer weights are loaded ONCE (excluded from both buckets,
 * already measured separately as native load latency). Each repeat then
 * re-initializes both sessions from scratch (handoff_init gives a clean,
 * unoccupied receiver every time) and times:
 *   fixed_cost_sender_plus_adapter = sender prefill of the 32-symbol
 *     sequence + handoff_import into the receiver, paid once per handoff.
 *   marginal_cost_cache_query = one receiver query ("Class:") against the
 *     already-imported cache.
 *
 * No native re-prefill counterpart exists to compare against (the
 * receiver's 6-layer geometry has no native from-raw-text CLI; see
 * paper.md Section 6/Appendix A), so only these two costs are reported,
 * not a breakeven query count or a cache-transfer-vs-reprefill comparison.
 */
#include "tensor_gpt2_handoff.h"
#include "tensor_gpt2_bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void gpt2_matmul_bias_accelerate(const float *, int, int, const float *, const float *, int, float *);

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000.0 + t.tv_nsec / 1e6; }

static int checked_path(char *out, size_t cap, const char *dir, const char *name) {
    int n = snprintf(out, cap, "%s/%s", dir, name); return n < 0 || (size_t)n >= cap ? -1 : 0;
}
static int fingerprint(const char *path, const char *sha) {
    char actual[65]; return strlen(sha) != 64 || sha256_file(path, actual) || strcmp(actual, sha);
}

static Gpt2Weights sender, receiver;
static HandoffSession source, target;
static HandoffBridge bridge;
static Gpt2Bpe tokenizer;
static float logits[GPT2_VOCAB];

static void stats(const double *v, int n, double *mn, double *mx, double *mean, double *median) {
    double sorted[4096]; for (int i = 0; i < n; i++) sorted[i] = v[i];
    for (int i = 1; i < n; i++) { double k = sorted[i]; int j = i - 1; while (j >= 0 && sorted[j] > k) { sorted[j+1] = sorted[j]; j--; } sorted[j+1] = k; }
    *mn = sorted[0]; *mx = sorted[n-1]; *median = sorted[n/2];
    double s = 0; for (int i = 0; i < n; i++) s += v[i]; *mean = s / n;
}

int main(int argc, char **argv) {
    if (argc != 7) { fprintf(stderr, "usage: native_amortized_latency ASSET_DIR SENDER_SHA RECEIVER_SHA BRIDGE_SHA WARMUP REPEATS\n"); return 2; }
    const char *dir = argv[1];
    int warmup = atoi(argv[5]), repeats = atoi(argv[6]);
    if (repeats < 1 || repeats > 4096 || warmup < 0) { fprintf(stderr, "bad warmup/repeats\n"); return 2; }

    char path[4096], vocab[4096], merges[4096];
    if (checked_path(path, sizeof path, dir, "sender.safetensors") || gpt2_load_weights_layers(&sender, path, argv[2], 12)) return 2;
    if (checked_path(path, sizeof path, dir, "receiver.safetensors") || gpt2_load_weights_layers(&receiver, path, argv[3], 6)) return 2;
    if (checked_path(path, sizeof path, dir, "bridge.safetensors") || fingerprint(path, argv[4]) || handoff_load_bridge(&bridge, path)) return 2;
    if (checked_path(vocab, sizeof vocab, dir, "vocab.json") || checked_path(merges, sizeof merges, dir, "merges.txt") || gpt2_bpe_load(&tokenizer, vocab, merges)) return 2;

    const char *sequence = "AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB";
    char text[100] = "Sequence:"; size_t p = strlen(text);
    for (int i = 0; i < 32; i++) { text[p++] = ' '; text[p++] = sequence[i]; }
    text[p] = 0;
    int ids[64]; int n = gpt2_bpe_encode(&tokenizer, text, (int)p, ids, 64);
    if (n <= 0 || n + 2 > 64) return 2;
    int query[16]; int nq = gpt2_bpe_encode(&tokenizer, "Class:", 6, query, 16);
    if (nq <= 0 || n + nq > MAXCACHE) return 2;

    int total = warmup + repeats;
    double *fixed_ms = malloc(sizeof(double) * repeats), *marginal_ms = malloc(sizeof(double) * repeats);
    if (!fixed_ms || !marginal_ms) return 2;

    for (int rep = 0; rep < total; rep++) {
        if (handoff_init(&source, &sender, 12, gpt2_matmul_bias_accelerate) || handoff_init(&target, &receiver, 6, gpt2_matmul_bias_accelerate)) return 2;

        double t0 = now_ms();
        for (int i = 0; i < n; i++) if (handoff_token(&source, ids[i], NULL)) return 2;
        if (handoff_import(&target, &source, &bridge)) return 2;
        double t1 = now_ms();

        for (int i = 0; i < nq; i++) if (handoff_token(&target, query[i], i == nq - 1 ? logits : NULL)) return 2;
        double t2 = now_ms();

        if (rep >= warmup) { fixed_ms[rep - warmup] = t1 - t0; marginal_ms[rep - warmup] = t2 - t1; }
    }

    double fmn, fmx, fmean, fmed, mmn, mmx, mmean, mmed;
    stats(fixed_ms, repeats, &fmn, &fmx, &fmean, &fmed);
    stats(marginal_ms, repeats, &mmn, &mmx, &mmean, &mmed);

    printf("{\n");
    printf("  \"mechanism\": \"native_amortized_latency\",\n");
    printf("  \"method\": \"Native FASM/Accelerate tensor-kv-handoff binary, in-process repeats after weight loading (excluded from both buckets, measured separately). Fixed cost = sender prefill of the 32-symbol sequence + handoff_import, paid once. Marginal cache-query cost = one receiver query against the already-imported cache. No native re-prefill counterpart exists to compare against.\",\n");
    printf("  \"warmup\": %d,\n  \"repeats\": %d,\n", warmup, repeats);
    printf("  \"fixed_cost_sender_plus_adapter\": {\"min_ms\": %.6f, \"max_ms\": %.6f, \"mean_ms\": %.6f, \"median_ms\": %.6f},\n", fmn, fmx, fmean, fmed);
    printf("  \"marginal_cost_cache_query\": {\"min_ms\": %.6f, \"max_ms\": %.6f, \"mean_ms\": %.6f, \"median_ms\": %.6f}\n", mmn, mmx, mmean, mmed);
    printf("}\n");
    return 0;
}
