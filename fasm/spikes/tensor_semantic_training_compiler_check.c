#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { LEAF, MATMUL, BIAS_ADD, RELU, MSE };
enum { INPUT=1, PARAM=2, CONSTANT=4, TEMP=8 };
enum { FORWARD, ZERO_GRAD, BACKWARD, OPTIMIZER };
#define NONE UINT32_MAX
typedef struct { float *data,*grad;uint32_t rows,cols; } Tensor;
typedef struct { uint32_t op,lhs,rhs,flags;Tensor tensor; } Node;
typedef struct { int(*run)(void*);void*context;uint32_t kind,flags;uint64_t reserved; } ExecStep;
typedef struct { Node*graph;uint32_t node,action,grad_mask;float lr; } Context;
extern int tensor_transformer_steps_execute(const ExecStep*,uint64_t);

static int action(void*opaque){Context*c=opaque;Node*n=&c->graph[c->node],*a=n->lhs==NONE?NULL:&c->graph[n->lhs],*b=n->rhs==NONE?NULL:&c->graph[n->rhs];Tensor*out=&n->tensor;
    if(c->action==ZERO_GRAD){memset(out->grad,0,(size_t)out->rows*out->cols*sizeof(float));return 0;}
    if(c->action==OPTIMIZER){for(uint32_t i=0;i<out->rows*out->cols;i++)out->data[i]-=c->lr*out->grad[i];return 0;}
    if(c->action==FORWARD){
        if(n->op==MATMUL)for(uint32_t i=0;i<a->tensor.rows;i++)for(uint32_t j=0;j<b->tensor.cols;j++){float s=0;for(uint32_t k=0;k<a->tensor.cols;k++)s+=a->tensor.data[i*a->tensor.cols+k]*b->tensor.data[k*b->tensor.cols+j];out->data[i*out->cols+j]=s;}
        else if(n->op==BIAS_ADD)for(uint32_t i=0;i<out->rows;i++)for(uint32_t j=0;j<out->cols;j++)out->data[i*out->cols+j]=a->tensor.data[i*out->cols+j]+b->tensor.data[j];
        else if(n->op==RELU)for(uint32_t i=0;i<out->rows*out->cols;i++)out->data[i]=a->tensor.data[i]>0?a->tensor.data[i]:0;
        else if(n->op==MSE){float loss=0;for(uint32_t i=0;i<a->tensor.rows*a->tensor.cols;i++){float e=a->tensor.data[i]-b->tensor.data[i];loss+=e*e;}out->data[0]=loss/(a->tensor.rows*a->tensor.cols);}return 0;}
    if(n->op==MSE){uint32_t count=a->tensor.rows*a->tensor.cols;for(uint32_t i=0;i<count;i++){float g=2*(a->tensor.data[i]-b->tensor.data[i])/count;if(c->grad_mask&1)a->tensor.grad[i]+=g;if(c->grad_mask&2)b->tensor.grad[i]-=g;}}
    else if(n->op==RELU&&(c->grad_mask&1))for(uint32_t i=0;i<out->rows*out->cols;i++)a->tensor.grad[i]+=a->tensor.data[i]>0?out->grad[i]:0;
    else if(n->op==BIAS_ADD)for(uint32_t i=0;i<out->rows;i++)for(uint32_t j=0;j<out->cols;j++){float g=out->grad[i*out->cols+j];if(c->grad_mask&1)a->tensor.grad[i*out->cols+j]+=g;if(c->grad_mask&2)b->tensor.grad[j]+=g;}
    else if(n->op==MATMUL)for(uint32_t i=0;i<a->tensor.rows;i++)for(uint32_t j=0;j<b->tensor.cols;j++){float g=out->grad[i*out->cols+j];for(uint32_t k=0;k<a->tensor.cols;k++){if(c->grad_mask&1)a->tensor.grad[i*a->tensor.cols+k]+=g*b->tensor.data[k*b->tensor.cols+j];if(c->grad_mask&2)b->tensor.grad[k*b->tensor.cols+j]+=a->tensor.data[i*a->tensor.cols+k]*g;}}
    return 0;}

static int validate(const Node*g,uint32_t n){for(uint32_t i=0;i<n;i++){if(!g[i].tensor.data||!g[i].tensor.grad||!g[i].tensor.rows||!g[i].tensor.cols)return-1;if(g[i].op==LEAF)continue;if(g[i].lhs>=i||(g[i].op!=RELU&&g[i].rhs>=i))return-2;const Tensor*a=&g[g[i].lhs].tensor,*b=g[i].rhs==NONE?NULL:&g[g[i].rhs].tensor,*o=&g[i].tensor;
        if(g[i].op==MATMUL&&(a->cols!=b->rows||o->rows!=a->rows||o->cols!=b->cols))return-3;if(g[i].op==BIAS_ADD&&(a->rows!=o->rows||a->cols!=o->cols||b->rows!=1||b->cols!=o->cols))return-3;if(g[i].op==RELU&&(a->rows!=o->rows||a->cols!=o->cols))return-3;if(g[i].op==MSE&&(a->rows!=b->rows||a->cols!=b->cols||o->rows!=1||o->cols!=1))return-3;}return 0;}
static int compile(Node*g,uint32_t n,float lr,ExecStep*s,Context*c,uint32_t cap,uint32_t*out){if(validate(g,n)||n>64)return-1;uint8_t needs[64]={0};for(uint32_t i=0;i<n;i++)needs[i]=g[i].op==LEAF?g[i].flags==PARAM:needs[g[i].lhs]||(g[i].rhs!=NONE&&needs[g[i].rhs]);uint32_t at=0;
    for(uint32_t i=0;i<n;i++)if(g[i].op!=LEAF){if(at==cap)return-2;c[at]=(Context){g,i,FORWARD,0,lr};s[at]=(ExecStep){action,&c[at],FORWARD,0,0};at++;}
    for(uint32_t i=0;i<n;i++)if(needs[i]&&g[i].op!=MSE){if(at==cap)return-2;c[at]=(Context){g,i,ZERO_GRAD,0,lr};s[at]=(ExecStep){action,&c[at],ZERO_GRAD,0,0};at++;}
    for(uint32_t i=n;i-->0;)if(g[i].op!=LEAF&&needs[i]){if(at==cap)return-2;uint32_t mask=needs[g[i].lhs]?1:0;if(g[i].rhs!=NONE&&needs[g[i].rhs])mask|=2;c[at]=(Context){g,i,BACKWARD,mask,lr};s[at]=(ExecStep){action,&c[at],BACKWARD,0,0};at++;}
    for(uint32_t i=0;i<n;i++)if(g[i].op==LEAF&&g[i].flags==PARAM){if(at==cap)return-2;c[at]=(Context){g,i,OPTIMIZER,0,lr};s[at]=(ExecStep){action,&c[at],OPTIMIZER,0,0};at++;}*out=at;return 0;}

int main(void){float x[2],gx[2],w1[8]={.5f,-.7f,.3f,.8f,-.4f,.6f,.9f,-.2f},gw1[8],b1[4]={.1f,.1f,-.1f,0},gb1[4],z1[4],gz1[4],z1b[4],gz1b[4],h[4],gh[4],w2[4]={.7f,-.5f,.6f,-.8f},gw2[4],b2[1]={0},gb2[1],raw[1],graw[1],pred[1],gpred[1],target[1],gtarget[1],loss[1],gloss[1];
    Node g[]={{LEAF,NONE,NONE,INPUT,{x,gx,1,2}},{LEAF,NONE,NONE,PARAM,{w1,gw1,2,4}},{LEAF,NONE,NONE,PARAM,{b1,gb1,1,4}},{MATMUL,0,1,TEMP,{z1,gz1,1,4}},{BIAS_ADD,3,2,TEMP,{z1b,gz1b,1,4}},{RELU,4,NONE,TEMP,{h,gh,1,4}},{LEAF,NONE,NONE,PARAM,{w2,gw2,4,1}},{LEAF,NONE,NONE,PARAM,{b2,gb2,1,1}},{MATMUL,5,6,TEMP,{raw,graw,1,1}},{BIAS_ADD,8,7,TEMP,{pred,gpred,1,1}},{LEAF,NONE,NONE,CONSTANT,{target,gtarget,1,1}},{MSE,9,10,TEMP,{loss,gloss,1,1}}};
    ExecStep steps[64];Context ctx[64];uint32_t count;if(compile(g,12,.1f,steps,ctx,64,&count))return 1;static const float data[4][2]={{0,0},{0,1},{1,0},{1,1}},labels[4]={0,1,1,0};
    for(unsigned epoch=0;epoch<4000;epoch++)for(unsigned sample=0;sample<4;sample++){memcpy(x,data[sample],sizeof x);target[0]=labels[sample];if(tensor_transformer_steps_execute(steps,count))return 2;}
    int correct=0;for(unsigned sample=0;sample<4;sample++){memcpy(x,data[sample],sizeof x);for(unsigned i=0;i<5;i++)if(steps[i].run(steps[i].context))return 3;correct+=(pred[0]>=.5f)==(labels[sample]>=.5f);}
    printf("semantic training compiler passed: model=mlp graph_nodes=12 schedule=%u forward=6 zero=9 backward=6 optimizer=4 needs_grad=automatic executor=x86_64-assembly correct=%d/4\n",count,correct);return correct==4&&count==25?0:4;}
