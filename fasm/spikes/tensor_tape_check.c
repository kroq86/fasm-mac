#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *data;
    float *grad;
    uint64_t rows;
    uint64_t cols;
} Tensor;

typedef struct {
    uint32_t op;
    uint32_t lhs;
    uint32_t rhs;
    uint32_t flags;
    Tensor *tensor;
    uint64_t reserved;
} TapeNode;

enum { OP_LEAF, OP_MATMUL, OP_RELU, OP_MSE };
enum { FLAG_PARAMETER=1, FLAG_CONSTANT=2, FLAG_INPUT=4, FLAG_TEMPORARY=8 };
#define NO_INDEX UINT32_MAX

extern int tensor_shape_validate_f32(const Tensor *, uint64_t *);
extern int tensor_tape_validate_f32(const TapeNode *, uint64_t);

static int check_shapes(void) {
    float storage = 0.0f;
    uint64_t elements = 0;
    Tensor valid = {&storage, NULL, 7, 9};
    if (tensor_shape_validate_f32(&valid, &elements) || elements != 63) return 1;
    Tensor zero = {&storage, NULL, 0, 9};
    if (tensor_shape_validate_f32(&zero, NULL) != -1) return 1;
    Tensor product_overflow = {&storage, NULL, UINT64_MAX, 2};
    if (tensor_shape_validate_f32(&product_overflow, NULL) != -2) return 1;
    Tensor byte_overflow = {&storage, NULL, UINT64_MAX / 4 + 1, 1};
    if (tensor_shape_validate_f32(&byte_overflow, NULL) != -2) return 1;
    return 0;
}

static int check_tape(void) {
    float storage[4] = {0};
    Tensor tensors[4] = {
        {storage, NULL, 1, 1}, {storage, NULL, 1, 1},
        {storage, NULL, 1, 1}, {storage, NULL, 1, 1}
    };
    TapeNode source[4] = {
        {OP_LEAF, NO_INDEX, NO_INDEX, FLAG_PARAMETER, &tensors[0], 0},
        {OP_LEAF, NO_INDEX, NO_INDEX, FLAG_INPUT, &tensors[1], 0},
        {OP_MATMUL, 0, 1, FLAG_TEMPORARY, &tensors[2], 0},
        {OP_RELU, 2, NO_INDEX, FLAG_TEMPORARY, &tensors[3], 0},
    };
    if (tensor_tape_validate_f32(source, 4)) return 1;

    TapeNode *relocated = malloc(sizeof(source) + 64);
    if (!relocated) return 1;
    memcpy((char *)relocated + 64, source, sizeof(source));
    TapeNode *copy = (TapeNode *)((char *)relocated + 64);
    int rc = tensor_tape_validate_f32(copy, 4);
    free(relocated);
    if (rc) return 1;

    source[2].lhs = 3;
    if (tensor_tape_validate_f32(source, 4) != -2) return 1;
    source[2].lhs = 0;
    source[3].op = 99;
    if (tensor_tape_validate_f32(source, 4) != -2) return 1;
    source[3].op = OP_RELU;
    source[0].flags = FLAG_PARAMETER | FLAG_CONSTANT;
    if (tensor_tape_validate_f32(source, 4) != -2) return 1;
    source[0].flags = FLAG_PARAMETER;
    source[2].flags = FLAG_PARAMETER;
    if (tensor_tape_validate_f32(source, 4) != -2) return 1;
    source[2].flags = FLAG_TEMPORARY;
    source[0].tensor = NULL;
    if (tensor_tape_validate_f32(source, 4) != -2) return 1;
    return 0;
}

int main(void) {
    if (check_shapes() || check_tape()) {
        fputs("tensor tape spike failed\n", stderr);
        return 1;
    }
    puts("tensor tape spike passed: indices/relocation/topology/overflow");
    return 0;
}
