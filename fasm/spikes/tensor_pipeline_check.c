#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { float *data, *grad; uint64_t rows, cols; } Tensor;
typedef struct { uint32_t op, lhs, rhs, flags; Tensor *tensor; uint32_t arena_slot,generation; } Node;
typedef struct { void *base; uint64_t capacity,used,generation; } Arena;
enum { OP_LEAF, OP_MATMUL, OP_RELU, OP_MSE };
enum { FLAG_PARAMETER=1, FLAG_CONSTANT=2, FLAG_INPUT=4, FLAG_TEMPORARY=8 };
#define NONE UINT32_MAX

extern int tensor_tape_forward_f32(Node *, uint64_t, Arena *const *,uint64_t);
extern int tensor_tape_backward_f32(Node *, uint64_t, Arena *const *,uint64_t);

int main(void) {
    float ad[6] = {0.4f,-0.7f,0.2f, 0.5f,0.3f,-0.6f}, ag[6] = {0};
    float bd[6] = {0.8f,-0.2f, -0.5f,0.9f, 0.7f,0.4f}, bg[6] = {0};
    float td[4] = {0.1f,0.4f,0.7f,0.2f}, tg[4] = {0};
    float md[4] = {0}, mg[4] = {0}, rd[4] = {0}, rg[4] = {0};
    float ld = 0, lg = 1;
    Tensor t[6] = {{ad,ag,2,3}, {bd,bg,3,2}, {td,tg,2,2},
                   {md,mg,2,2}, {rd,rg,2,2}, {&ld,&lg,1,1}};
    Arena persistent={0,0,0,1},scratch={0,0,0,1};
    Arena *arenas[2]={&persistent,&scratch};
    Node tape[6] = {{OP_LEAF,NONE,NONE,FLAG_PARAMETER,&t[0],0,1},
                    {OP_LEAF,NONE,NONE,FLAG_PARAMETER,&t[1],0,1},
                    {OP_LEAF,NONE,NONE,FLAG_CONSTANT,&t[2],0,1},
                    {OP_MATMUL,0,1,FLAG_TEMPORARY,&t[3],1,1},
                    {OP_RELU,3,NONE,FLAG_TEMPORARY,&t[4],1,1},
                    {OP_MSE,4,2,FLAG_TEMPORARY,&t[5],1,1}};
    int rc = tensor_tape_forward_f32(tape, 6, arenas, 2);
    if (rc) { fprintf(stderr, "initial forward rc=%d\n", rc); return 1; }
    rc = tensor_tape_backward_f32(tape, 6, arenas, 2);
    if (rc) { fprintf(stderr, "initial backward rc=%d\n", rc); return 1; }

    const float eps = 0.001f, tol = 0.0002f;
    for (int group = 0; group < 2; ++group) {
        float *data = group ? bd : ad, *grad = group ? bg : ag;
        int count = 6;
        for (int i = 0; i < count; ++i) {
            float saved = data[i];
            data[i] = saved + eps; if (tensor_tape_forward_f32(tape,6,arenas,2)) return 1; float plus=ld;
            data[i] = saved - eps; if (tensor_tape_forward_f32(tape,6,arenas,2)) return 1; float minus=ld;
            data[i] = saved;
            float numeric = (plus-minus)/(2*eps);
            if (fabsf(numeric-grad[i]) > tol) {
                fprintf(stderr,"pipeline grad mismatch group=%d i=%d analytic=%g numeric=%g\n",
                        group,i,grad[i],numeric);
                return 1;
            }
        }
    }
    memset(ag,0,sizeof ag); memset(bg,0,sizeof bg); memset(tg,0,sizeof tg);
    memset(mg,0,sizeof mg); memset(rg,0,sizeof rg);
    rc = tensor_tape_forward_f32(tape,6,arenas,2);
    if (rc) { fprintf(stderr, "accumulation forward rc=%d\n", rc); return 1; }
    rc = tensor_tape_backward_f32(tape,6,arenas,2);
    if (rc) { fprintf(stderr, "accumulation backward rc=%d\n", rc); return 1; }
    float first = ag[0];
    rc = tensor_tape_backward_f32(tape,6,arenas,2);
    if (rc) { fprintf(stderr, "second backward rc=%d\n", rc); return 1; }
    if (fabsf(ag[0]-2*first) > 1e-6f) {
        fprintf(stderr, "second backward did not accumulate: first=%g now=%g\n", first, ag[0]);
        return 1;
    }
    float before=md[0]; scratch.generation=2;
    if(tensor_tape_forward_f32(tape,6,arenas,2)!=-4 || md[0]!=before){
        fputs("stale lifetime token reached a kernel\n",stderr);return 1;
    }
    puts("tensor pipeline spike passed: matmul/relu/mse/tape/backward/lifetime/finite-difference");
    return 0;
}
