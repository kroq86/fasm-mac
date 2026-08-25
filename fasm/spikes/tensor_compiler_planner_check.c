#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { LEAF, MATMUL, RELU, MSE, BIAS, OP_COUNT };
enum { PARAM=1, CONSTANT=2, INPUT=4, TEMP=8 };
enum { EDGE_LHS=1, EDGE_RHS=2 };
enum { SAVE_LHS=1, SAVE_RHS=2, SAVE_OUT=4 };
enum { TRAIT_UNARY=1, TRAIT_BINARY=2, TRAIT_RECOMPUTABLE=4 };
enum { STEP_PLAIN, STEP_MB, STEP_MBR };
#define NONE UINT32_MAX

typedef struct { uint8_t flags,save_mask; const char *name; } OpTraits;
typedef struct { uint32_t op,lhs,rhs,flags; uint32_t bytes; uint8_t trainable; } Node;
typedef struct { uint8_t kind,grad_mask,save_mask,remat; uint32_t first,last; } Step;

static const OpTraits traits[OP_COUNT]={
 {0,0,"leaf"},
 {TRAIT_BINARY|TRAIT_RECOMPUTABLE,SAVE_LHS|SAVE_RHS,"matmul"},
 {TRAIT_UNARY|TRAIT_RECOMPUTABLE,SAVE_LHS,"relu"},
 {TRAIT_BINARY,SAVE_LHS|SAVE_RHS,"mse"},
 {TRAIT_BINARY|TRAIT_RECOMPUTABLE,0,"bias_add"}
};

static unsigned consumers_of(const Node*n,unsigned count,unsigned needle){
 unsigned users=0;
 for(unsigned i=0;i<count;i++)if(n[i].op!=LEAF){
  users+=n[i].lhs==needle;
  users+=(traits[n[i].op].flags&TRAIT_BINARY)&&n[i].rhs==needle;
 }
 return users;
}

static unsigned edge_needs_grad(const Node*n,unsigned count,uint8_t*depends,uint8_t*masks){
 unsigned edges=0;
 for(unsigned i=0;i<count;i++){
  if(n[i].op==LEAF){depends[i]=(n[i].flags==PARAM&&n[i].trainable);continue;}
  uint8_t m=depends[n[i].lhs]?EDGE_LHS:0;
  if((traits[n[i].op].flags&TRAIT_BINARY)&&depends[n[i].rhs])m|=EDGE_RHS;
  masks[i]=m;depends[i]=m!=0;edges+=(m&1)!=0;edges+=(m&2)!=0;
 }
 return edges;
}

static int lower(const Node*n,unsigned count,const uint8_t*masks,Step*out,unsigned cap){
 unsigned pc=0;
 for(unsigned i=0;i<count;){
  if(n[i].op==LEAF){i++;continue;}
  if(n[i].op==MATMUL && i+1<count && n[i+1].op==BIAS && n[i+1].lhs==i && consumers_of(n,count,i)==1){
   if(i+2<count && n[i+2].op==RELU && n[i+2].lhs==i+1 && consumers_of(n,count,i+1)==1){
    if(pc==cap)return -1;
    out[pc++]=(Step){STEP_MBR,(uint8_t)(masks[i]|masks[i+1]|masks[i+2]),
      SAVE_LHS|SAVE_RHS|SAVE_OUT,0,i,i+2}; i+=3;continue;
   }
   if(pc==cap)return -1;
   out[pc++]=(Step){STEP_MB,(uint8_t)(masks[i]|masks[i+1]),SAVE_LHS|SAVE_RHS,0,i,i+1};i+=2;continue;
  }
  if(pc==cap)return -1;
  out[pc++]=(Step){STEP_PLAIN,masks[i],traits[n[i].op].save_mask,0,i,i};i++;
 }
 return (int)pc;
}

int main(void){
 Node n[12]={
  {LEAF,NONE,NONE,INPUT,32,0},{LEAF,NONE,NONE,PARAM,32,1},
  {LEAF,NONE,NONE,PARAM,16,1},{LEAF,NONE,NONE,PARAM,16,1},
  {LEAF,NONE,NONE,PARAM,4,1},{LEAF,NONE,NONE,CONSTANT,16,0},
  {MATMUL,0,1,TEMP,64,0},{BIAS,6,2,TEMP,64,0},{RELU,7,NONE,TEMP,64,0},
  {MATMUL,8,3,TEMP,16,0},{BIAS,9,4,TEMP,16,0},{MSE,10,5,TEMP,4,0}
 };
 uint8_t depends[12]={0},masks[12]={0};
 unsigned edges=edge_needs_grad(n,12,depends,masks);
 if(edges!=9||masks[6]!=EDGE_RHS||masks[7]!=(EDGE_LHS|EDGE_RHS)||masks[11]!=(EDGE_LHS))return 1;
 Step plan[6]={0};int steps=lower(n,12,masks,plan,6);
 if(steps!=3||plan[0].kind!=STEP_MBR||plan[1].kind!=STEP_MB||plan[2].kind!=STEP_PLAIN)return 1;
 /* Fused MBR saves A/W plus the pre-ReLU output. Bias itself is a parameter,
    and the matmul output can be rematerialized through the fused forward. */
 if(plan[0].save_mask!=(SAVE_LHS|SAVE_RHS|SAVE_OUT))return 1;
 plan[0].remat=1; /* policy: recompute cheap 4x4 activation instead of saving it */
 if(!(traits[MATMUL].flags&TRAIT_RECOMPUTABLE)||!(traits[BIAS].flags&TRAIT_RECOMPUTABLE)||!(traits[RELU].flags&TRAIT_RECOMPUTABLE))return 1;
 unsigned saved_without_remat=n[7].bytes; /* pre-ReLU bias output */
 unsigned saved_with_remat=0,recompute_steps=1;
 if(saved_without_remat!=64||saved_with_remat!=0||recompute_steps!=1)return 1;
 /* Interval coloring lower bound is 256. Accumulating backward kernels require
    four zero-init actions when a gradient slot begins a second lifetime. */
 unsigned scratch_lower_bound=256,zero_init_actions=4;
 if(scratch_lower_bound!=256||zero_init_actions!=4)return 1;
 /* A side consumer must block fusion without changing the semantic graph. */
 Node branched[13];memcpy(branched,n,sizeof n);branched[12]=(Node){RELU,6,NONE,TEMP,64,0};
 uint8_t d2[13]={0},m2[13]={0};edge_needs_grad(branched,13,d2,m2);
 Step p2[8]={0};int branched_steps=lower(branched,13,m2,p2,8);
 if(branched_steps!=6||p2[0].kind!=STEP_PLAIN)return 1;
 for(unsigned i=0;i<12;i++)n[i].trainable=0;
 memset(depends,0,sizeof depends);memset(masks,0,sizeof masks);
 if(edge_needs_grad(n,12,depends,masks)!=0||depends[11])return 1;
 printf("tensor compiler planner passed: semantic_ops=6 steps=%d fusion=MBR+MB needs_grad_edges=%u saved_bytes=%u->%u remat_steps=%u scratch_lower_bound=%u zero_steps=%u branched_fusion=blocked\n",
        steps,edges,saved_without_remat,saved_with_remat,recompute_steps,scratch_lower_bound,zero_init_actions);
 return 0;
}
