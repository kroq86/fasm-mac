#ifndef TENSOR_SAMPLING_H
#define TENSOR_SAMPLING_H
/* Decoder-runtime Etap 2, operation 6/10: top-k/temperature sampling.
 * Not a canonical-graph op -- sampling from a probability distribution is
 * not a differentiable operation, so this is a standalone, explicitly
 * seeded, replayable utility (same seed -> same draw, matching the
 * project's existing determinism/replay discipline elsewhere), not a
 * tensor_semantic_compiler.h op.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

/* xorshift32, deterministic and seedable -- not cryptographic, just needs
 * to be reproducible given the same explicit state, matching the same
 * "explicit state capsule" convention the neurosymbolic executor line
 * already established this session. */
typedef struct { uint32_t state; } SampleRng;
static uint32_t sample_rng_next(SampleRng *r) {
    uint32_t x = r->state;
    /* xorshift32 has an absorbing all-zero state. Normalize it so every
     * serialized 32-bit seed denotes a usable deterministic stream. */
    if (x == 0) x = UINT32_C(0x6d2b79f5);
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    r->state = x;
    return x;
}
static float sample_rng_uniform01(SampleRng *r) { return (float)(sample_rng_next(r) >> 8) / (float)(1u << 24); } /* [0,1) */

/* Greedy: argmax over logits. temperature=0 is defined to mean this. */
static int sample_argmax(const float *logits, int vocab) {
    if (!logits || vocab <= 0 || vocab > (1 << 16)) return -1;
    for (int i = 0; i < vocab; i++) if (!isfinite(logits[i])) return -1;
    int best = 0;
    for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

/* Top-k + temperature sampling: scale logits by 1/temperature, restrict
 * to the k highest, softmax over just those (numerically stable, same
 * max-subtraction convention as k_softmax_rows_fwd), sample from the
 * resulting categorical distribution via inverse-CDF on one uniform draw.
 * k<=0 or k>=vocab means "no restriction" (full-vocab sampling).
 * temperature<=0 falls back to sample_argmax (matches how real decoding
 * CLIs treat temperature=0). Returns the sampled token id. */
static int sample_top_k_temperature(const float *logits, int vocab, int k, float temperature, SampleRng *rng) {
    if (!logits || !rng || vocab <= 0 || vocab > (1 << 16) || !isfinite(temperature)) return -1;
    for (int i = 0; i < vocab; i++) if (!isfinite(logits[i])) return -1;
    if (temperature <= 0.0f) return sample_argmax(logits, vocab);
    if (k <= 0 || k >= vocab) {
        /* Full-vocabulary softmax needs no selection scratch and therefore
         * must not inherit the bounded top-k workspace limit. */
        float mx = logits[0] / temperature;
        for (int i = 1; i < vocab; i++) { float z = logits[i] / temperature; if (z > mx) mx = z; }
        double total = 0;
        for (int i = 0; i < vocab; i++) total += exp((double)logits[i] / temperature - mx);
        double target = (double)sample_rng_uniform01(rng) * total, cumulative = 0;
        for (int i = 0; i < vocab; i++) { cumulative += exp((double)logits[i] / temperature - mx); if (target < cumulative) return i; }
        return vocab - 1;
    }
    if (k > 256) return -1; /* explicit bounded-workspace contract */

    /* find the k highest logits (their vocab indices), via k passes of
     * "find max among not-yet-taken" -- vocab/k here are always small in
     * this project's test/toy scale, O(k*vocab) is fine and simple */
    int idx[256];
    int kk = k;
    for (int r = 0; r < kk; r++) {
        int best = -1;
        for (int i = 0; i < vocab; i++) {
            int used = 0; for (int q = 0; q < r; q++) used |= idx[q] == i;
            if (!used && (best < 0 || logits[i] > logits[best])) best = i;
        }
        idx[r] = best;
    }

    float scaled[256], mx = -INFINITY;
    for (int r = 0; r < kk; r++) { scaled[r] = logits[idx[r]] / temperature; if (scaled[r] > mx) mx = scaled[r]; }
    float total = 0, prob[256];
    for (int r = 0; r < kk; r++) { prob[r] = expf(scaled[r] - mx); total += prob[r]; }
    for (int r = 0; r < kk; r++) prob[r] /= total;

    float u = sample_rng_uniform01(rng);
    float cum = 0;
    for (int r = 0; r < kk; r++) { cum += prob[r]; if (u < cum) return idx[r]; }
    return idx[kk - 1]; /* floating-point tail: cum may land just under 1.0 */
}

#endif
