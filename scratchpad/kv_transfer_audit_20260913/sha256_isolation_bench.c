#include <stdio.h>
#include <time.h>
#include "tensor_sha256.h"

int main(int argc, char **argv) {
    char hex[65];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 1; i < argc; i++) {
        if (sha256_file(argv[i], hex)) { fprintf(stderr, "failed: %s\n", argv[i]); return 1; }
        printf("%s  %s\n", hex, argv[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("total sha256_file time: %.1f ms\n", ms);
    return 0;
}
