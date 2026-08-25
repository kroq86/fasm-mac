#include <math.h>
#include <stdint.h>
#include <stdio.h>

typedef struct { float *data,*grad; uint64_t rows,cols; } Tensor;
typedef struct { uint32_t op,lhs,rhs,flags; Tensor *tensor; uint64_t reserved; } Node;
enum { PARAMETER=1, CONSTANT=2, INPUT=4, TEMPORARY=8 };
extern int tensor_parameters_sgd_f32(Node *,uint64_t,float);
extern int tensor_parameters_zero_grad_f32(Node *,uint64_t);

int main(void) {
    float pd[2]={1.0f,-2.0f}, pg[2]={0.5f,-0.25f};
    float qd[1]={3.0f}, qg[1]={2.0f};
    float cd[1]={9.0f}, id[1]={8.0f}, td[1]={7.0f};
    Tensor p={pd,pg,1,2}, q={qd,qg,1,1}, c={cd,0,1,1}, in={id,0,1,1}, tmp={td,0,1,1};
    Node nodes[5]={{0,0,0,PARAMETER,&p,0},{0,0,0,CONSTANT,&c,0},
                   {0,0,0,INPUT,&in,0},{1,0,0,TEMPORARY,&tmp,0},
                   {0,0,0,PARAMETER,&q,0}};
    if (tensor_parameters_sgd_f32(nodes,5,0.1f)) return 1;
    if (fabsf(pd[0]-0.95f)>1e-6f || fabsf(pd[1]+1.975f)>1e-6f || fabsf(qd[0]-2.8f)>1e-6f) return 1;
    if (cd[0]!=9.0f || id[0]!=8.0f || td[0]!=7.0f) return 1;
    if (tensor_parameters_zero_grad_f32(nodes,5)) return 1;
    if (pg[0]!=0 || pg[1]!=0 || qg[0]!=0) return 1;

    pg[0]=1; pg[1]=1; qg[0]=1;
    nodes[4].tensor=&p;
    float before=pd[0];
    if (tensor_parameters_sgd_f32(nodes,5,0.1f)!=-2) return 1;
    if (pd[0]!=before) { fputs("duplicate preflight allowed partial update\n",stderr); return 1; }
    puts("tensor SGD spike passed: parameters-only/preflight/dedup/zero-grad");
    return 0;
}
