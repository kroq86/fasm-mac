#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { void *base; uint64_t capacity, used, generation; } Arena;
typedef struct {
    float *data, *grad;
    uint64_t rows, cols;
    Arena *owner;
    uint64_t generation;
} OwnedTensor;

extern int tensor_arena_init_f32(Arena *, void *, uint64_t);
extern int tensor_arena_reset_f32(Arena *);
extern int tensor_arena_alloc_f32(Arena *, OwnedTensor *, uint64_t, uint64_t, uint64_t);
extern int tensor_owned_validate_f32(const OwnedTensor *);

int main(void) {
    void *persistent_mem = malloc(4096), *scratch_mem = malloc(4096);
    if (!persistent_mem || !scratch_mem) return 1;
    Arena persistent, scratch;
    if (tensor_arena_init_f32(&persistent, persistent_mem, 4096)) return 1;
    if (tensor_arena_init_f32(&scratch, scratch_mem, 4096)) return 1;
    OwnedTensor parameter, temporary, replacement;
    if (tensor_arena_alloc_f32(&persistent, &parameter, 3, 2, 1)) return 1;
    if (tensor_arena_alloc_f32(&scratch, &temporary, 2, 2, 1)) return 1;
    if (((uintptr_t)parameter.data & 63) || ((uintptr_t)parameter.grad & 63)) return 1;
    if (((uintptr_t)temporary.data & 63) || ((uintptr_t)temporary.grad & 63)) return 1;
    if (tensor_owned_validate_f32(&parameter) || tensor_owned_validate_f32(&temporary)) return 1;
    parameter.data[0] = 7.0f;
    void *old_temporary_data = temporary.data;
    if (tensor_arena_reset_f32(&scratch)) return 1;
    if (tensor_owned_validate_f32(&temporary) != -4) return 1;
    if (tensor_owned_validate_f32(&parameter) || parameter.data[0] != 7.0f) return 1;
    if (tensor_arena_alloc_f32(&scratch, &replacement, 2, 2, 1)) return 1;
    if (replacement.data != old_temporary_data) return 1;
    OwnedTensor too_large;
    if (tensor_arena_alloc_f32(&scratch, &too_large, 1024, 1024, 1) != -3) return 1;
    if (tensor_arena_alloc_f32(&scratch, &too_large, UINT64_MAX, 2, 1) != -2) return 1;
    free(persistent_mem); free(scratch_mem);
    puts("tensor arena spike passed: alignment/persistent/scratch/reset/stale/overflow");
    return 0;
}
