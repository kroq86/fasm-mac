#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float *data,*grad; uint64_t rows,cols; } Tensor;
extern int tensor_matmul_forward_f32(const Tensor*,const Tensor*,Tensor*);
extern int tensor_matmul_backward_f32(Tensor*,Tensor*,const Tensor*);

static int transpose_forward(const Tensor *in,Tensor *out){
 if(!in||!out||!in->data||!out->data||out->rows!=in->cols||out->cols!=in->rows)return -1;
 for(uint64_t i=0;i<in->rows;i++)for(uint64_t j=0;j<in->cols;j++)out->data[j*out->cols+i]=in->data[i*in->cols+j];
 return 0;
}
static int transpose_backward(Tensor *in,const Tensor *out){
 if(!in->grad||!out->grad||out->rows!=in->cols||out->cols!=in->rows)return -1;
 for(uint64_t i=0;i<in->rows;i++)for(uint64_t j=0;j<in->cols;j++)in->grad[i*in->cols+j]+=out->grad[j*out->cols+i];
 return 0;
}
static int softmax_rows_forward(const Tensor *in,Tensor *out){
 if(!in||!out||!in->data||!out->data||in->rows!=out->rows||in->cols!=out->cols)return -1;
 for(uint64_t i=0;i<in->rows;i++){
  float max=in->data[i*in->cols];for(uint64_t j=1;j<in->cols;j++)if(in->data[i*in->cols+j]>max)max=in->data[i*in->cols+j];
  float sum=0;for(uint64_t j=0;j<in->cols;j++){float e=expf(in->data[i*in->cols+j]-max);out->data[i*out->cols+j]=e;sum+=e;}
  for(uint64_t j=0;j<in->cols;j++)out->data[i*out->cols+j]/=sum;
 }
 return 0;
}
static int softmax_rows_backward(Tensor *in,const Tensor *out){
 if(!in->grad||!out->grad||in->rows!=out->rows||in->cols!=out->cols)return -1;
 for(uint64_t i=0;i<in->rows;i++){
  float dot=0;for(uint64_t j=0;j<in->cols;j++)dot+=out->grad[i*out->cols+j]*out->data[i*out->cols+j];
  for(uint64_t j=0;j<in->cols;j++)in->grad[i*in->cols+j]+=out->data[i*out->cols+j]*(out->grad[i*out->cols+j]-dot);
 }
 return 0;
}
static float seed(uint64_t i,uint64_t salt){return (float)((int)((i*37+salt*19)%29)-14)/31.0f;}
static int closef(float a,float b,float tolerance){return fabsf(a-b)<=tolerance*fmaxf(1.0f,fmaxf(fabsf(a),fabsf(b)));}

typedef struct {
 uint64_t tokens,features,dim;
 Tensor x,wq,wk,wv,q,k,v,kt,scores,weights,out;
 float scale;
} Attention;

static int forward(Attention *a){
 if(tensor_matmul_forward_f32(&a->x,&a->wq,&a->q)||tensor_matmul_forward_f32(&a->x,&a->wk,&a->k)||
    tensor_matmul_forward_f32(&a->x,&a->wv,&a->v)||transpose_forward(&a->k,&a->kt)||
    tensor_matmul_forward_f32(&a->q,&a->kt,&a->scores))return -1;
 for(uint64_t i=0;i<a->tokens*a->tokens;i++)a->scores.data[i]*=a->scale;
 if(softmax_rows_forward(&a->scores,&a->weights)||tensor_matmul_forward_f32(&a->weights,&a->v,&a->out))return -1;
 return 0;
}
static int backward(Attention *a){
 if(tensor_matmul_backward_f32(&a->weights,&a->v,&a->out)||softmax_rows_backward(&a->scores,&a->weights))return -1;
 for(uint64_t i=0;i<a->tokens*a->tokens;i++)a->scores.grad[i]*=a->scale;
 if(tensor_matmul_backward_f32(&a->q,&a->kt,&a->scores)||transpose_backward(&a->k,&a->kt)||
    tensor_matmul_backward_f32(&a->x,&a->wv,&a->v)||tensor_matmul_backward_f32(&a->x,&a->wk,&a->k)||
    tensor_matmul_backward_f32(&a->x,&a->wq,&a->q))return -1;
 return 0;
}
static float objective(Attention *a,const float *seed_grad){
 if(forward(a))return NAN;float sum=0;for(uint64_t i=0;i<a->tokens*a->dim;i++)sum+=a->out.data[i]*seed_grad[i];return sum;
}
static void scalar_reference(const Attention *a,float *out){
 float *q=calloc(a->tokens*a->dim,sizeof(float)),*k=calloc(a->tokens*a->dim,sizeof(float)),*v=calloc(a->tokens*a->dim,sizeof(float));
 float *weights=calloc(a->tokens*a->tokens,sizeof(float));if(!q||!k||!v||!weights)exit(2);
 for(uint64_t i=0;i<a->tokens;i++)for(uint64_t j=0;j<a->dim;j++)for(uint64_t p=0;p<a->features;p++){
  float x=a->x.data[i*a->features+p];q[i*a->dim+j]+=x*a->wq.data[p*a->dim+j];k[i*a->dim+j]+=x*a->wk.data[p*a->dim+j];v[i*a->dim+j]+=x*a->wv.data[p*a->dim+j];
 }
 for(uint64_t i=0;i<a->tokens;i++){
  float max=-INFINITY;for(uint64_t j=0;j<a->tokens;j++){float s=0;for(uint64_t p=0;p<a->dim;p++)s+=q[i*a->dim+p]*k[j*a->dim+p];weights[i*a->tokens+j]=s*a->scale;if(weights[i*a->tokens+j]>max)max=weights[i*a->tokens+j];}
  float sum=0;for(uint64_t j=0;j<a->tokens;j++){weights[i*a->tokens+j]=expf(weights[i*a->tokens+j]-max);sum+=weights[i*a->tokens+j];}
  for(uint64_t j=0;j<a->tokens;j++)weights[i*a->tokens+j]/=sum;
 }
 for(uint64_t i=0;i<a->tokens;i++)for(uint64_t j=0;j<a->dim;j++)for(uint64_t p=0;p<a->tokens;p++)out[i*a->dim+j]+=weights[i*a->tokens+p]*v[p*a->dim+j];
 free(q);free(k);free(v);free(weights);
}
static float *floats(size_t n){float *p=calloc(n,sizeof *p);if(!p){perror("calloc");exit(2);}return p;}
static Tensor tensor(uint64_t rows,uint64_t cols,int grad){Tensor t={floats(rows*cols),grad?floats(rows*cols):0,rows,cols};return t;}
static void clear_grad(Tensor *t){if(t->grad)memset(t->grad,0,t->rows*t->cols*sizeof(float));}

static int one(uint64_t tokens,uint64_t features,uint64_t dim){
 Attention a={.tokens=tokens,.features=features,.dim=dim,.scale=1.0f/sqrtf((float)dim)};
 a.x=tensor(tokens,features,1);a.wq=tensor(features,dim,1);a.wk=tensor(features,dim,1);a.wv=tensor(features,dim,1);
 a.q=tensor(tokens,dim,1);a.k=tensor(tokens,dim,1);a.v=tensor(tokens,dim,1);a.kt=tensor(dim,tokens,1);
 a.scores=tensor(tokens,tokens,1);a.weights=tensor(tokens,tokens,1);a.out=tensor(tokens,dim,1);
 for(uint64_t i=0;i<tokens*features;i++)a.x.data[i]=seed(i,1)*.4f;
 for(uint64_t i=0;i<features*dim;i++){a.wq.data[i]=seed(i,2)*.3f;a.wk.data[i]=seed(i,3)*.3f;a.wv.data[i]=seed(i,4)*.3f;}
 float *out_seed=floats(tokens*dim);for(uint64_t i=0;i<tokens*dim;i++)out_seed[i]=seed(i,5);
 if(forward(&a))return 1;
 float *reference=floats(tokens*dim);scalar_reference(&a,reference);
 for(uint64_t i=0;i<tokens*dim;i++)if(!closef(a.out.data[i],reference[i],2e-5f))return 1;
 for(uint64_t i=0;i<tokens;i++){float sum=0;for(uint64_t j=0;j<tokens;j++)sum+=a.weights.data[i*tokens+j];if(!closef(sum,1,2e-6f))return 1;}
 memcpy(a.out.grad,out_seed,tokens*dim*sizeof(float));if(backward(&a))return 1;
 struct Probe{Tensor *t;uint64_t at;} probes[]={{&a.x,tokens*features/2},{&a.wq,features*dim/3},{&a.wk,features*dim/2},{&a.wv,features*dim*2/3}};
 for(uint64_t p=0;p<sizeof probes/sizeof probes[0];p++){
  Tensor *t=probes[p].t;uint64_t at=probes[p].at;float analytic=t->grad[at],saved=t->data[at],eps=1e-3f;
  t->data[at]=saved+eps;float plus=objective(&a,out_seed);t->data[at]=saved-eps;float minus=objective(&a,out_seed);t->data[at]=saved;
  float numeric=(plus-minus)/(2*eps);if(!closef(analytic,numeric,3e-3f)){fprintf(stderr,"attention grad mismatch shape=%llux%llux%llu probe=%llu analytic=%g numeric=%g\n",tokens,features,dim,p,analytic,numeric);return 1;}
 }
 Tensor *all[]={&a.x,&a.wq,&a.wk,&a.wv,&a.q,&a.k,&a.v,&a.kt,&a.scores,&a.weights,&a.out};
 for(uint64_t i=0;i<sizeof all/sizeof all[0];i++){clear_grad(all[i]);free(all[i]->data);free(all[i]->grad);}free(out_seed);free(reference);
 return 0;
}

int main(void){
 const uint64_t dims[]={7,8,9};unsigned cases=0;
 for(unsigned t=0;t<3;t++)for(unsigned f=0;f<3;f++)for(unsigned d=0;d<3;d++){if(one(dims[t],dims[f],dims[d]))return 1;cases++;}
 printf("tensor attention spike passed: cases=%u tokens/features/dim=7/8/9 qkv=matmul kt=materialized softmax=stable forward=scalar-reference backward=finite-difference abi=2d-only\n",cases);
 return 0;
}
