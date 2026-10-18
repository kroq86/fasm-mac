/* Rejection boundaries of the canonical compiler, not a general graph fuzzer.
 * Unsupported schedules are inspected, never executed (including pre-fix).
 */
#define M 1
#define H 1
#define D 1
#define QW 3
#define MAXCACHE 2
#include "tensor_semantic_compiler.h"
#include <stdio.h>

static unsigned failures, checks;

static void check(int ok, const char *name) {
    checks++;
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    failures += !ok;
}

static void reject(Node *g, uint32_t n, const char *name) {
    ExecStep steps[32], before_steps[32];
    Context contexts[32], before_contexts[32];
    uint32_t count = 777;
    memset(steps, 0xa5, sizeof steps);
    memset(contexts, 0x5a, sizeof contexts);
    memcpy(before_steps, steps, sizeof steps);
    memcpy(before_contexts, contexts, sizeof contexts);
    int rc = compile(g, n, .01f, steps, contexts, 32, &count);
    int ok = rc == -1 && count == 777 &&
             !memcmp(steps, before_steps, sizeof steps) &&
             !memcmp(contexts, before_contexts, sizeof contexts);
    check(ok, name);
}

typedef struct {
    float qkv[3], gqkv[3], keys[2], gkeys[2], values[2];
    float output, grad, unrelated, gunrelated;
    Node graph[4];
} Cached;

static void cached_init(Cached *f) {
    memset(f, 0, sizeof *f);
    f->qkv[0] = 1; f->qkv[1] = 2; f->qkv[2] = 3;
    f->unrelated = 7;
    f->graph[0] = (Node){LEAF, NONE, NONE, INPUT,
        {f->qkv, f->gqkv, NULL, 1, 3, 0}};
    f->graph[1] = (Node){LEAF, NONE, NONE, INPUT,
        {f->keys, f->gkeys, f->values, 2, 1, 0}};
    f->graph[2] = (Node){CAUSAL_ATTENTION_CACHED, 0, 1, TEMP,
        {&f->output, &f->grad, NULL, 1, 1, 0}};
    f->graph[3] = (Node){LEAF, NONE, NONE, PARAM,
        {&f->unrelated, &f->gunrelated, NULL, 1, 1, 0}};
}

static void cached_execute(Cached *f, uint32_t n, const char *name) {
    ExecStep steps[32]; Context contexts[32]; uint32_t count = 0;
    int rc = compile(f->graph, n, .01f, steps, contexts, 32, &count);
    if (!rc) rc = tensor_transformer_steps_execute(steps, count);
    check(!rc && f->output == 3 && f->keys[0] == 2 && f->values[0] == 3 &&
          f->graph[1].tensor.aux_count == 1 && f->unrelated == 7, name);
}

int main(void) {
    float a = 2, b = 3, output = 0, ga = 0, gb = 0, go = 0;
    Node dense[3] = {
        {LEAF, NONE, NONE, INPUT, {&a, &ga, NULL, 1, 1, 0}},
        {LEAF, NONE, NONE, INPUT, {&b, &gb, NULL, 1, 1, 0}},
        {MATMUL, 0, 1, TEMP, {&output, &go, NULL, 1, 1, 0}},
    };
    ExecStep steps[8]; Context contexts[8]; uint32_t count = 0;
    int rc = compile(dense, 3, .01f, steps, contexts, 8, &count);
    if (!rc) rc = tensor_transformer_steps_execute(steps, count);
    check(!rc && count == 1 && output == 6, "valid_matmul_executes");
    dense[2].tensor.cols = 2;
    check(validate(dense, 3) == -3, "bad_shape_control");
    reject(dense, 3, "bad_shape_rejected_before_emission");
    dense[2].tensor.cols = 1;
    const uint32_t invalid_ops[] = {OP_COUNT, UINT32_MAX};
    for (unsigned i = 0; i < sizeof invalid_ops / sizeof invalid_ops[0]; i++) {
        dense[2].op = invalid_ops[i];
        check(validate(dense, 3) != 0, i ? "uint_max_validate" : "op_count_validate");
        reject(dense, 3, i ? "uint_max_compile" : "op_count_compile");
    }

    Cached f;
    cached_init(&f);
    cached_execute(&f, 3, "cached_inference_executes");
    cached_init(&f);
    cached_execute(&f, 4, "unrelated_param_preserves_cached_inference");

    const struct { unsigned node, flags; const char *name; } cases[] = {
        {0, PARAM, "qkv_param"},
        {0, INPUT | RETAIN_GRAD, "qkv_retain_grad"},
        {1, INPUT | RETAIN_GRAD, "cache_retain_grad"},
        {1, PARAM, "cache_param"},
        {2, TEMP | RETAIN_GRAD, "cached_output_retain_grad"},
    };
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        cached_init(&f);
        f.graph[cases[i].node].flags = cases[i].flags;
        if (cases[i].node == 1 && cases[i].flags == PARAM) {
            /* Keep the existing PARAM aux-metadata validation satisfied;
             * this target must fail for missing BWD, not invalid metadata. */
            f.values[0] = 1;
            f.graph[1].tensor.aux_count = 1;
        }
        check(validate(f.graph, 3) == 0, "gradient_target_shape_control");
        reject(f.graph, 3, cases[i].name);
    }

    cached_init(&f);
    float intermediate[3] = {0}, gintermediate[3] = {0};
    Node chain[4] = {
        f.graph[0],
        {RELU, 0, NONE, TEMP, {intermediate, gintermediate, NULL, 1, 3, 0}},
        f.graph[1], f.graph[2],
    };
    chain[0].flags = PARAM;
    chain[3].lhs = 1; chain[3].rhs = 2;
    check(validate(chain, 4) == 0, "transitive_param_shape_control");
    reject(chain, 4, "transitive_param_rejected");

    printf("compiler rejection check: checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
