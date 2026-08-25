#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { float *data, *grad; uint64_t rows, cols; } TensorView;
typedef struct {
    float *data, *grad; uint64_t rows, cols;
    void *owner; uint64_t generation;
} OwnedTensor;
typedef struct {
    uint32_t op, lhs, rhs, flags;
    TensorView *tensor;
    uint32_t arena_slot, generation;
} TokenNode;

int main(void) {
    _Static_assert(sizeof(TensorView) == 32, "TensorView ABI changed");
    _Static_assert(sizeof(OwnedTensor) == 48, "OwnedTensor ABI changed");
    _Static_assert(sizeof(TokenNode) == 32, "token node no longer fits 32 bytes");

    float reused[4] = {0};
    TensorView stale = {reused, NULL, 2, 2};
    TensorView fresh = {reused, NULL, 2, 2};
    if (memcmp(&stale, &fresh, sizeof(stale)) != 0) return 1;

    TokenNode stale_node = {0,UINT32_MAX,UINT32_MAX,4,&stale,3,10};
    TokenNode fresh_node = {0,UINT32_MAX,UINT32_MAX,4,&fresh,3,11};
    uint32_t arena_generation = 11;
    if (stale_node.generation == arena_generation) return 1;
    if (fresh_node.generation != arena_generation) return 1;

    printf("lifetime comparison: owned=%zu view=%zu token_node=%zu ",
           sizeof(OwnedTensor),sizeof(TensorView),sizeof(TokenNode));
    puts("result=view-alone-cannot-detect-reuse tape-token-can");
    return 0;
}
