#include <stdint.h>
#include <stdio.h>
#include <string.h>
enum{QKV,VIEW,ATTENTION,CONTIGUOUS,MATMUL,RESIDUAL,LAYERNORM,RELU,ZERO_GRAD,REMAT};
typedef struct{int(*run)(void*);void*context;uint32_t kind,flags;uint64_t reserved;}ExecStep;
typedef struct{uint32_t kind,generation,*slot_generation,*trace;uint32_t*trace_count,fail_kind;}Context;
_Static_assert(sizeof(ExecStep)==32,"experimental ExecStep ABI");
extern int tensor_transformer_steps_execute(const ExecStep*,uint64_t);
static int action(void*opaque){Context*c=opaque;if(*c->slot_generation!=c->generation)return-4;c->trace[(*c->trace_count)++]=c->kind;return c->kind==c->fail_kind?-7:0;}
static unsigned emit(const uint32_t*kinds,unsigned count,ExecStep*steps,Context*contexts,uint32_t*generation,uint32_t*trace,uint32_t*trace_count,uint32_t fail){for(unsigned i=0;i<count;i++){contexts[i]=(Context){kinds[i],*generation,generation,trace,trace_count,fail};steps[i]=(ExecStep){action,&contexts[i],kinds[i],0,0};}return count;}
int main(void){
 static const uint32_t forward[]={QKV,VIEW,VIEW,VIEW,ATTENTION,CONTIGUOUS,MATMUL,RESIDUAL,LAYERNORM,MATMUL,RELU,MATMUL,RESIDUAL,LAYERNORM};
 static const uint32_t reverse_compute[]={LAYERNORM,RESIDUAL,MATMUL,RELU,MATMUL,LAYERNORM,RESIDUAL,MATMUL,CONTIGUOUS,ATTENTION,QKV};
 uint32_t backward[21],at=0;for(unsigned i=0;i<9;i++)backward[at++]=ZERO_GRAD;for(unsigned i=0;i<8;i++)backward[at++]=reverse_compute[i];backward[at++]=REMAT;for(unsigned i=8;i<11;i++)backward[at++]=reverse_compute[i];if(at!=21)return 1;
 ExecStep steps[21];Context contexts[21];uint32_t generation=7,trace[64],trace_count=0;
 emit(forward,14,steps,contexts,&generation,trace,&trace_count,UINT32_MAX);if(tensor_transformer_steps_execute(steps,14)||trace_count!=14||memcmp(trace,forward,sizeof forward))return 1;
 trace_count=0;emit(backward,21,steps,contexts,&generation,trace,&trace_count,UINT32_MAX);if(tensor_transformer_steps_execute(steps,21)||trace_count!=21||memcmp(trace,backward,sizeof backward))return 1;
 trace_count=0;emit(forward,14,steps,contexts,&generation,trace,&trace_count,ATTENTION);if(tensor_transformer_steps_execute(steps,14)!=-7||trace_count!=5)return 1;
 trace_count=0;emit(forward,14,steps,contexts,&generation,trace,&trace_count,UINT32_MAX);generation++;if(tensor_transformer_steps_execute(steps,14)!=-4||trace_count)return 1;
 printf("tensor transformer executor passed: ExecStep=32 forward=14 backward=21 direct_call=yes view=3 contiguous=1 remat=1 zero_grad=9 error_stop=yes stale=detected abi=experimental\n");return 0;
}
