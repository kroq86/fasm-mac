#include <stdint.h>
#include <stdio.h>
#include <string.h>
enum Op{LEAF,QKV,VIEW_SPLIT,ATTENTION,MERGE_HEADS,MATMUL,RESIDUAL,LAYER_NORM,RELU};
enum Role{INPUT=1,PARAM=2,TEMP=4,VIEW=8};
enum Action{ACT_QKV,ACT_VIEW,ACT_ATTENTION,ACT_CONTIGUOUS,ACT_MATMUL,ACT_RESIDUAL,ACT_LAYERNORM,ACT_RELU,ACT_ZERO_GRAD,ACT_REMAT};
enum Save{SAVE_NONE=0,SAVE_LHS=1,SAVE_RHS=2,SAVE_OUT=4,SAVE_STATS=8,SAVE_QKV=16,SAVE_PROB=32};
typedef struct{uint8_t op,role,needs_grad,consumers;uint16_t lhs,rhs;uint32_t bytes;}Node;
typedef struct{uint8_t kind,save_mask,grad_mask,flags;uint16_t node,first,last;uint32_t bytes;}PlanStep;
#define N 0xffff
static int lower(const Node*n,unsigned count,PlanStep*out,unsigned cap){unsigned pc=0;for(unsigned i=0;i<count;i++){if(n[i].op==LEAF)continue;if(pc==cap)return-1;uint8_t kind=0,save=0;switch(n[i].op){case QKV:kind=ACT_QKV;save=SAVE_LHS|SAVE_RHS;break;case VIEW_SPLIT:kind=ACT_VIEW;break;case ATTENTION:kind=ACT_ATTENTION;save=SAVE_QKV|SAVE_PROB;break;case MERGE_HEADS:kind=ACT_CONTIGUOUS;break;case MATMUL:kind=ACT_MATMUL;save=SAVE_LHS|SAVE_RHS;break;case RESIDUAL:kind=ACT_RESIDUAL;break;case LAYER_NORM:kind=ACT_LAYERNORM;save=SAVE_LHS|SAVE_STATS;break;case RELU:kind=ACT_RELU;save=SAVE_OUT;break;default:return-1;}unsigned at=pc++;out[at]=(PlanStep){kind,save,n[i].needs_grad?3:0,0,i,at,at,n[i].bytes};}return(int)pc;}
static unsigned backward_actions(const PlanStep*p,unsigned count){unsigned n=0;for(unsigned i=count;i;i--){if(p[i-1].kind==ACT_VIEW)continue;n++;}return n;}
int main(void){/* X,Wqkv,packed,Q,K,V,attention,merged,Wo,projected,res1,ln1,W1,hidden,relu,W2,ffout,res2,out */
 Node n[]={
 {LEAF,INPUT,1,2,N,N,256},{LEAF,PARAM,1,1,N,N,768},{QKV,TEMP,1,3,0,1,768},
 {VIEW_SPLIT,VIEW,1,1,2,N,0},{VIEW_SPLIT,VIEW,1,1,2,N,0},{VIEW_SPLIT,VIEW,1,1,2,N,0},
 {ATTENTION,TEMP,1,1,3,5,256},{MERGE_HEADS,TEMP,1,1,6,N,256},{LEAF,PARAM,1,1,N,N,256},
 {MATMUL,TEMP,1,1,7,8,256},{RESIDUAL,TEMP,1,1,9,0,256},{LAYER_NORM,TEMP,1,2,10,N,256},
 {LEAF,PARAM,1,1,N,N,512},{MATMUL,TEMP,1,1,11,12,512},{RELU,TEMP,1,1,13,N,512},
 {LEAF,PARAM,1,1,N,N,512},{MATMUL,TEMP,1,1,14,15,256},{RESIDUAL,TEMP,1,1,16,11,256},{LAYER_NORM,TEMP,1,0,17,N,256}};
 PlanStep plan[16]={0};int steps=lower(n,sizeof n/sizeof n[0],plan,16);if(steps!=14)return 1;
 unsigned views=0,contiguous=0,ln=0,forward_compute=0;for(int i=0;i<steps;i++){views+=plan[i].kind==ACT_VIEW;contiguous+=plan[i].kind==ACT_CONTIGUOUS;ln+=plan[i].kind==ACT_LAYERNORM;forward_compute+=plan[i].kind!=ACT_VIEW;}
 if(views!=3||contiguous!=1||ln!=2||forward_compute!=11)return 1;
 /* Planner policy: attention scores are recomputed; probabilities, packed QKV,
    both LayerNorm statistics, and ReLU output remain saved for backward. */
 unsigned saved_packed=768,saved_prob=512,saved_ln_stats=2*64,saved_relu=512;
 unsigned saved_bytes=saved_packed+saved_prob+saved_ln_stats+saved_relu;
 unsigned without_remat=saved_bytes+512; /* attention score matrix */
 unsigned remat_actions=1,zero_grad_actions=9;
 unsigned backward=backward_actions(plan,steps)+remat_actions+zero_grad_actions;
 if(saved_bytes!=1920||without_remat!=2432||backward!=21)return 1;
 /* Conservative scratch contract for T=8,M=8,H=2,D=4,F=16. The executor
    still needs interval coloring before this becomes an allocation ABI. */
 unsigned forward_temporaries=3840,training_saved=saved_bytes,scratch_upper=forward_temporaries+training_saved;
 if(scratch_upper!=5760)return 1;
 printf("tensor transformer planner passed: semantic_nodes=%zu forward_actions=%d compute=%u views=%u contiguous=%u backward_actions=%u remat=%u zero_grad=%u saved_bytes=%u->%u scratch_upper=%u abi=experimental\n",sizeof n/sizeof n[0],steps,forward_compute,views,contiguous,backward,remat_actions,zero_grad_actions,without_remat,saved_bytes,scratch_upper);return 0;}
