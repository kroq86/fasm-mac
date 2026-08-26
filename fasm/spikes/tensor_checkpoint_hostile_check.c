#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { VALID, BAD_COUNT, BAD_RANK, BAD_DIMS, BAD_RANGE, BAD_OVERLAP, BAD_NAME };
enum { MAX_TENSORS = 1024, MAX_RANK = 8 };
static const uint64_t MAX_PAYLOAD = UINT64_C(1) << 34; /* 16 GiB policy ceiling */

typedef struct {
    char name[16];
    uint32_t rank, element_bytes;
    uint64_t dims[MAX_RANK], offset, bytes;
} Descriptor;

static int add_overflow(uint64_t a, uint64_t b, uint64_t *out) {
    *out = a + b;
    return *out < a;
}

static int mul_overflow(uint64_t a, uint64_t b, uint64_t *out) {
    if (a && b > UINT64_MAX / a) return 1;
    *out = a * b;
    return 0;
}

/* Metadata-only preflight: no allocation and no payload pointer is exposed
   until every descriptor has passed these checks. */
static int preflight(const Descriptor *d, uint64_t count, uint64_t file_bytes) {
    if (!count || count > MAX_TENSORS) return BAD_COUNT;
    if (file_bytes > MAX_PAYLOAD) return BAD_RANGE;
    for (uint64_t i = 0; i < count; i++) {
        if (!memchr(d[i].name, '\0', sizeof d[i].name) || !d[i].name[0]) return BAD_NAME;
        for (uint64_t j = i + 1; j < count; j++)
            if (!strncmp(d[i].name, d[j].name, sizeof d[i].name)) return BAD_NAME;
        if (!d[i].rank || d[i].rank > MAX_RANK) return BAD_RANK;
        if (d[i].element_bytes != 4 || d[i].offset % d[i].element_bytes) return BAD_RANGE;
        uint64_t elements = 1;
        for (uint32_t r = 0; r < d[i].rank; r++) {
            if (!d[i].dims[r] || mul_overflow(elements, d[i].dims[r], &elements)) return BAD_DIMS;
        }
        uint64_t computed, end;
        if (mul_overflow(elements, d[i].element_bytes, &computed) || computed != d[i].bytes) return BAD_DIMS;
        if (add_overflow(d[i].offset, d[i].bytes, &end) || end > file_bytes) return BAD_RANGE;
        for (uint64_t j = 0; j < i; j++) {
            uint64_t other_end;
            if (add_overflow(d[j].offset, d[j].bytes, &other_end)) return BAD_RANGE;
            if (d[i].offset < other_end && d[j].offset < end) return BAD_OVERLAP;
        }
    }
    return VALID;
}

static Descriptor good(const char *name, uint64_t offset) {
    Descriptor d = {.rank=2,.element_bytes=4,.dims={3,4},.offset=offset,.bytes=48};
    strncpy(d.name, name, sizeof d.name - 1);
    return d;
}

static uint64_t random64(uint64_t *state) {
    *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17;
    return *state;
}

int main(void) {
    Descriptor base[] = {good("wq", 256), good("wo", 304)};
    if (preflight(base, 2, 352) != VALID) return 1;
    if (preflight(base, UINT32_MAX, 352) != BAD_COUNT) return 2;
    Descriptor rank = base[0]; rank.rank = 900;
    if (preflight(&rank, 1, 352) != BAD_RANK) return 3;
    Descriptor product = base[0]; product.dims[0] = UINT64_MAX; product.dims[1] = 2;
    if (preflight(&product, 1, UINT64_MAX) != BAD_RANGE && preflight(&product, 1, MAX_PAYLOAD) != BAD_DIMS) return 4;
    Descriptor range = base[0]; range.offset = UINT64_MAX - 8;
    if (preflight(&range, 1, 352) != BAD_RANGE) return 5;
    Descriptor overlap[] = {good("a",256),good("b",288)};
    if (preflight(overlap, 2, 352) != BAD_OVERLAP) return 6;
    Descriptor duplicate[] = {good("same",256),good("same",304)};
    if (preflight(duplicate, 2, 352) != BAD_NAME) return 7;
    Descriptor unterminated = good("x",256); memset(unterminated.name, 'x', sizeof unterminated.name);
    if (preflight(&unterminated, 1, 352) != BAD_NAME) return 8;
    Descriptor misaligned = good("x",257);
    if (preflight(&misaligned, 1, 352) != BAD_RANGE) return 9;
    uint64_t seed = 0x726573756d6521ull;
    for (unsigned trial = 0; trial < 100000; trial++) {
        Descriptor fuzz = good("fuzz", random64(&seed));
        fuzz.rank = (uint32_t)(random64(&seed) % 12);
        fuzz.element_bytes = (uint32_t)(random64(&seed) % 9);
        fuzz.bytes = random64(&seed);
        for (unsigned i = 0; i < MAX_RANK; i++) fuzz.dims[i] = random64(&seed);
        (void)preflight(&fuzz, 1, random64(&seed) & (MAX_PAYLOAD - 1));
    }
    puts("tensor checkpoint hostile spike passed: preallocation=yes tensor_count/rank/dim_overflow/range/overlap/alignment/duplicate/unterminated_name=rejected metadata_fuzz=100000 max_payload=16GiB");
    return 0;
}
