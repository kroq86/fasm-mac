#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float *data,*grad; uint64_t rows,cols; } TensorView;
typedef struct { uint64_t dims[4],strides[4]; uint32_t rank,flags,arena_slot,generation; } TensorLayout;
typedef struct { TensorView *storage; TensorLayout *layout; } TensorRef;
_Static_assert(sizeof(TensorView)==32,"TensorView ABI");
_Static_assert(sizeof(TensorLayout)==80,"TensorLayout ABI");

static float *allocf(uint64_t n){float *p=calloc(n,sizeof(float));if(!p)exit(2);return p;}
static TensorLayout dense4(uint64_t b,uint64_t h,uint64_t m,uint64_t n){TensorLayout l={{b,h,m,n},{h*m*n,m*n,n,1},4,0,0,1};return l;}
static TensorLayout transpose_last2(TensorLayout l){uint64_t x=l.dims[2];l.dims[2]=l.dims[3];l.dims[3]=x;x=l.strides[2];l.strides[2]=l.strides[3];l.strides[3]=x;return l;}
static uint64_t at(const TensorLayout *l,uint64_t b,uint64_t h,uint64_t i,uint64_t j){return b*l->strides[0]+h*l->strides[1]+i*l->strides[2]+j*l->strides[3];}

static int matmul4_forward(const TensorRef *a,const TensorRef *b,TensorRef *o){
 TensorLayout *x=a->layout,*y=b->layout,*z=o->layout;
 if(x->rank!=4||y->rank!=4||z->rank!=4||x->dims[0]!=y->dims[0]||x->dims[1]!=y->dims[1]||x->dims[3]!=y->dims[2]||z->dims[2]!=x->dims[2]||z->dims[3]!=y->dims[3])return -1;
 for(uint64_t q=0;q<x->dims[0];q++)for(uint64_t h=0;h<x->dims[1];h++)for(uint64_t i=0;i<x->dims[2];i++)for(uint64_t j=0;j<y->dims[3];j++){
  float sum=0;for(uint64_t k=0;k<x->dims[3];k++)sum+=a->storage->data[at(x,q,h,i,k)]*b->storage->data[at(y,q,h,k,j)];o->storage->data[at(z,q,h,i,j)]=sum;
 }return 0;
}
static int matmul4_backward(TensorRef *a,TensorRef *b,const TensorRef *o){
 TensorLayout *x=a->layout,*y=b->layout,*z=o->layout;if(!a->storage->grad||!b->storage->grad||!o->storage->grad)return -1;
 for(uint64_t q=0;q<x->dims[0];q++)for(uint64_t h=0;h<x->dims[1];h++)for(uint64_t i=0;i<x->dims[2];i++)for(uint64_t j=0;j<y->dims[3];j++){
  float g=o->storage->grad[at(z,q,h,i,j)];for(uint64_t k=0;k<x->dims[3];k++){uint64_t ai=at(x,q,h,i,k),bi=at(y,q,h,k,j);a->storage->grad[ai]+=g*b->storage->data[bi];b->storage->grad[bi]+=g*a->storage->data[ai];}
 }return 0;
}
static int softmax4_forward(const TensorRef *in,TensorRef *out){
 TensorLayout *x=in->layout,*y=out->layout;uint64_t cols=x->dims[3];
 for(uint64_t b=0;b<x->dims[0];b++)for(uint64_t h=0;h<x->dims[1];h++)for(uint64_t i=0;i<x->dims[2];i++){
  float max=in->storage->data[at(x,b,h,i,0)];for(uint64_t j=1;j<cols;j++)max=fmaxf(max,in->storage->data[at(x,b,h,i,j)]);
  float sum=0;for(uint64_t j=0;j<cols;j++){float v=expf(in->storage->data[at(x,b,h,i,j)]-max);out->storage->data[at(y,b,h,i,j)]=v;sum+=v;}
  for(uint64_t j=0;j<cols;j++)out->storage->data[at(y,b,h,i,j)]/=sum;
 }return 0;
}
static int softmax4_backward(TensorRef *in,const TensorRef *out){
 TensorLayout *x=in->layout,*y=out->layout;uint64_t cols=x->dims[3];
 for(uint64_t b=0;b<x->dims[0];b++)for(uint64_t h=0;h<x->dims[1];h++)for(uint64_t i=0;i<x->dims[2];i++){
  float dot=0;for(uint64_t j=0;j<cols;j++)dot+=out->storage->grad[at(y,b,h,i,j)]*out->storage->data[at(y,b,h,i,j)];
  for(uint64_t j=0;j<cols;j++)in->storage->grad[at(x,b,h,i,j)]+=out->storage->data[at(y,b,h,i,j)]*(out->storage->grad[at(y,b,h,i,j)]-dot);
 }return 0;
}
typedef struct {TensorView qv,kv,vv,sv,wv,ov;TensorLayout ql,kl,ktl,sl,wl,ol;TensorRef q,k,kt,v,s,w,o;float scale;} Attention;
static int forward(Attention *a){if(matmul4_forward(&a->q,&a->kt,&a->s))return -1;uint64_t n=a->sl.dims[0]*a->sl.dims[1]*a->sl.dims[2]*a->sl.dims[3];for(uint64_t i=0;i<n;i++)a->sv.data[i]*=a->scale;return softmax4_forward(&a->s,&a->w)||matmul4_forward(&a->w,&a->v,&a->o);}
static int backward(Attention *a){if(matmul4_backward(&a->w,&a->v,&a->o)||softmax4_backward(&a->s,&a->w))return -1;uint64_t n=a->sl.dims[0]*a->sl.dims[1]*a->sl.dims[2]*a->sl.dims[3];for(uint64_t i=0;i<n;i++)a->sv.grad[i]*=a->scale;return matmul4_backward(&a->q,&a->kt,&a->s);}
static float seed(uint64_t i,uint64_t salt){return (float)((int)((i*41+salt*23)%37)-18)/29.0f;}
static float objective(Attention *a,const float *g){if(forward(a))return NAN;uint64_t n=a->ol.dims[0]*a->ol.dims[1]*a->ol.dims[2]*a->ol.dims[3];float sum=0;for(uint64_t i=0;i<n;i++)sum+=a->ov.data[i]*g[i];return sum;}
static int closef(float a,float b,float tol){return fabsf(a-b)<=tol*fmaxf(1.0f,fmaxf(fabsf(a),fabsf(b)));}

static int one(uint64_t tokens,uint64_t dim,uint64_t *saved){
 const uint64_t batches=2,heads=3,qd=batches*heads*tokens*dim,sd=batches*heads*tokens*tokens;
 Attention a={0};a.ql=dense4(batches,heads,tokens,dim);a.kl=dense4(batches,heads,tokens,dim);a.ktl=transpose_last2(a.kl);a.sl=dense4(batches,heads,tokens,tokens);a.wl=a.sl;a.ol=a.ql;a.scale=1/sqrtf((float)dim);
 a.qv=(TensorView){allocf(qd),allocf(qd),batches*heads*tokens,dim};a.kv=(TensorView){allocf(qd),allocf(qd),batches*heads*tokens,dim};a.vv=(TensorView){allocf(qd),allocf(qd),batches*heads*tokens,dim};
 a.sv=(TensorView){allocf(sd),allocf(sd),batches*heads*tokens,tokens};a.wv=(TensorView){allocf(sd),allocf(sd),batches*heads*tokens,tokens};a.ov=(TensorView){allocf(qd),allocf(qd),batches*heads*tokens,dim};
 a.q=(TensorRef){&a.qv,&a.ql};a.k=(TensorRef){&a.kv,&a.kl};a.kt=(TensorRef){&a.kv,&a.ktl};a.v=(TensorRef){&a.vv,&a.kl};a.s=(TensorRef){&a.sv,&a.sl};a.w=(TensorRef){&a.wv,&a.wl};a.o=(TensorRef){&a.ov,&a.ol};
 for(uint64_t i=0;i<qd;i++){a.qv.data[i]=seed(i,1)*.3f;a.kv.data[i]=seed(i,2)*.3f;a.vv.data[i]=seed(i,3)*.3f;a.ov.grad[i]=seed(i,4);}
 if(forward(&a))return 1;
 for(uint64_t b=0;b<batches;b++)for(uint64_t h=0;h<heads;h++)for(uint64_t i=0;i<tokens;i++){float sum=0;for(uint64_t j=0;j<tokens;j++)sum+=a.wv.data[at(&a.wl,b,h,i,j)];if(!closef(sum,1,2e-6f))return 1;}
 float *g=allocf(qd);memcpy(g,a.ov.grad,qd*sizeof(float));if(backward(&a))return 1;
 TensorView *probev[]={&a.qv,&a.kv,&a.vv};for(unsigned p=0;p<3;p++){uint64_t pos=qd*(p+1)/4;float analytic=probev[p]->grad[pos],old=probev[p]->data[pos],eps=1e-3f;probev[p]->data[pos]=old+eps;float plus=objective(&a,g);probev[p]->data[pos]=old-eps;float minus=objective(&a,g);probev[p]->data[pos]=old;float numeric=(plus-minus)/(2*eps);if(!closef(analytic,numeric,4e-3f)){fprintf(stderr,"multihead gradient mismatch t=%llu d=%llu probe=%u analytic=%g numeric=%g\n",tokens,dim,p,analytic,numeric);return 1;}}
 if(a.kt.storage!=a.k.storage||a.ktl.strides[2]!=1||a.ktl.strides[3]!=dim)return 1;*saved+=qd*sizeof(float);
 TensorView *all[]={&a.qv,&a.kv,&a.vv,&a.sv,&a.wv,&a.ov};for(unsigned i=0;i<6;i++){free(all[i]->data);free(all[i]->grad);}free(g);return 0;
}
int main(void){const uint64_t dims[]={7,8,9};uint64_t saved=0;unsigned cases=0;for(unsigned t=0;t<3;t++)for(unsigned d=0;d<3;d++){if(one(dims[t],dims[d],&saved))return 1;cases++;}printf("tensor multihead attention passed: cases=%u shape=[2,3,T,D] T/D=7/8/9 kt=strided-view softmax=stable backward=finite-difference avoided_kt_bytes=%llu\n",cases,(unsigned long long)saved);return 0;}
