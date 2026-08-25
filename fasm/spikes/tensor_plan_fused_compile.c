#include <stdint.h>
#include <stddef.h>

typedef struct { float *data,*grad; uint64_t rows,cols; } Tensor;
typedef struct { uint32_t op,lhs,rhs,flags; Tensor *tensor; uint32_t slot,gen; } Node;
typedef struct { void *forward,*backward; Tensor *lhs,*rhs,*out; } Step;
typedef struct { Tensor *a,*w,*bias,*matmul_out,*bias_out,*out; } FusionContext;
enum { LEAF, MATMUL, RELU, MSE, BIAS };

extern int tensor_matmul_forward_f32(Tensor*,Tensor*,Tensor*);
extern int tensor_matmul_backward_f32(Tensor*,Tensor*,Tensor*);
extern int tensor_relu_forward_f32(Tensor*,Tensor*);
extern int tensor_relu_backward_f32(Tensor*,Tensor*);
extern int tensor_mse_forward_f32(Tensor*,Tensor*,Tensor*);
extern int tensor_mse_backward_f32(Tensor*,Tensor*,Tensor*);
extern int tensor_bias_add_forward_f32(Tensor*,Tensor*,Tensor*);
extern int tensor_bias_add_backward_f32(Tensor*,Tensor*,Tensor*);
extern int tensor_plan_matmul_bias_forward_f32(void*,void*,void*);
extern int tensor_plan_matmul_bias_backward_f32(void*,void*,void*);
extern int tensor_plan_matmul_bias_relu_forward_f32(void*,void*,void*);
extern int tensor_plan_matmul_bias_relu_backward_f32(void*,void*,void*);

static unsigned consumers(const Node *n,uint64_t count,uint32_t target){
 unsigned result=0;
 for(uint64_t i=0;i<count;i++)if(n[i].op!=LEAF){
  result+=n[i].lhs==target;
  result+=(n[i].op!=RELU)&&(n[i].rhs==target);
 }
 return result;
}
static void plain_step(const Node*base,const Node*node,Step*s){
 s->lhs=base[node->lhs].tensor;s->rhs=node->op==RELU?0:base[node->rhs].tensor;s->out=node->tensor;
 switch(node->op){
  case MATMUL:s->forward=(void*)tensor_matmul_forward_f32;s->backward=(void*)tensor_matmul_backward_f32;break;
  case RELU:s->forward=(void*)tensor_relu_forward_f32;s->backward=(void*)tensor_relu_backward_f32;break;
  case MSE:s->forward=(void*)tensor_mse_forward_f32;s->backward=(void*)tensor_mse_backward_f32;break;
  default:s->forward=(void*)tensor_bias_add_forward_f32;s->backward=(void*)tensor_bias_add_backward_f32;break;
 }
}

/* Validated topological graph in, fused PlanSteps and plan-owned contexts out. */
int tensor_plan_compile_fused_f32(const Node*n,uint64_t count,Step*steps,uint64_t cap,
                                  FusionContext*contexts,uint64_t context_cap,
                                  uint64_t*out_count,uint64_t*out_context_count){
 if(!n||!count||!steps||!contexts||!out_count||!out_context_count)return -1;
 uint64_t pc=0,cc=0;
 for(uint64_t i=0;i<count;){
  if(n[i].op==LEAF){i++;continue;}
  if(n[i].op==MATMUL&&i+1<count&&n[i+1].op==BIAS&&n[i+1].lhs==i&&consumers(n,count,(uint32_t)i)==1){
   int with_relu=i+2<count&&n[i+2].op==RELU&&n[i+2].lhs==i+1&&consumers(n,count,(uint32_t)(i+1))==1;
   if(pc==cap||cc==context_cap)return -3;
   FusionContext*c=&contexts[cc++];
   c->a=n[n[i].lhs].tensor;c->w=n[n[i].rhs].tensor;c->bias=n[n[i+1].rhs].tensor;
   c->matmul_out=n[i].tensor;c->bias_out=n[i+1].tensor;c->out=with_relu?n[i+2].tensor:n[i+1].tensor;
   steps[pc].lhs=(Tensor*)c;steps[pc].rhs=0;steps[pc].out=c->out;
   if(with_relu){steps[pc].forward=(void*)tensor_plan_matmul_bias_relu_forward_f32;steps[pc].backward=(void*)tensor_plan_matmul_bias_relu_backward_f32;i+=3;}
   else{steps[pc].forward=(void*)tensor_plan_matmul_bias_forward_f32;steps[pc].backward=(void*)tensor_plan_matmul_bias_backward_f32;i+=2;}
   pc++;continue;
  }
  if(pc==cap)return -3;
  if(n[i].op<MATMUL||n[i].op>BIAS)return -2;
  plain_step(n,&n[i],&steps[pc++]);i++;
 }
 *out_count=pc;*out_context_count=cc;return 0;
}
