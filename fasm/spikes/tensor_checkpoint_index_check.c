#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { FOUND = 0, MISSING = -1, DUPLICATE = -2, SHAPE = -3, MAX = 8 };
typedef struct { char name[16]; uint32_t count; const float *data; } Entry;
typedef struct { char name[16]; uint32_t count; float *data; int required; } Request;

static int validate_names(const Entry *entries, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (!memchr(entries[i].name, '\0', sizeof entries[i].name)) return DUPLICATE;
        for (uint32_t j = i + 1; j < n; j++)
            if (!strncmp(entries[i].name, entries[j].name, sizeof entries[i].name)) return DUPLICATE;
    }
    return FOUND;
}

static int load_named(const Entry *entries, uint32_t n, Request *requests, uint32_t wanted) {
    if (n > MAX || wanted > MAX || validate_names(entries, n)) return DUPLICATE;
    for (uint32_t r = 0; r < wanted; r++) {
        const Entry *match = NULL;
        for (uint32_t i = 0; i < n; i++)
            if (!strncmp(entries[i].name, requests[r].name, sizeof entries[i].name)) match = &entries[i];
        if (!match) { if (requests[r].required) return MISSING; continue; }
        if (match->count != requests[r].count) return SHAPE;
        memcpy(requests[r].data, match->data, match->count * sizeof(float));
    }
    return FOUND;
}

int main(void) {
    const float wq[] = {1,2,3,4}, wo[] = {5,6}, future[] = {7};
    Entry reordered[] = {{"future",1,future},{"wo",2,wo},{"wq",4,wq}};
    float got_wq[4] = {0}, got_wo[2] = {0}, optional[3] = {0};
    Request wanted[] = {{"wq",4,got_wq,1},{"wo",2,got_wo,1},{"optimizer",3,optional,0}};
    if (load_named(reordered, 3, wanted, 3) || memcmp(wq, got_wq, sizeof wq) || memcmp(wo, got_wo, sizeof wo)) return 1;
    Entry duplicate[] = {{"wq",4,wq},{"wq",4,wq}};
    if (load_named(duplicate, 2, wanted, 3) != DUPLICATE) return 2;
    Entry missing[] = {{"wq",4,wq}};
    if (load_named(missing, 1, wanted, 3) != MISSING) return 3;
    Entry wrong_shape[] = {{"wq",2,wq},{"wo",2,wo}};
    if (load_named(wrong_shape, 2, wanted, 3) != SHAPE) return 4;
    puts("tensor checkpoint index spike passed: reordered=loaded unknown=skipped optional_missing=allowed duplicate=rejected required_missing=rejected shape_mismatch=rejected decision=name-index-required");
    return 0;
}
