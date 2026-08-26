#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float *data,*grad; uint64_t rows,cols; } TensorView;
typedef struct {
 uint64_t dims[4],strides[4];
 uint32_t rank,flags,arena_slot,generation;
} TensorLayout;
typedef struct { TensorView *storage; TensorLayout *layout; } TensorRef;
typedef struct { uint32_t generation; } ArenaSlot;

_Static_assert(sizeof(TensorView)==32,"legacy TensorView ABI changed");
_Static_assert(sizeof(TensorLayout)==80,"layout metadata size changed");
_Static_assert(sizeof(TensorRef)==16,"TensorRef size changed");

static int valid(const TensorRef *ref,const ArenaSlot *slots,uint32_t count){
 if(!ref||!ref->storage||!ref->layout||!ref->storage->data)return -1;
 TensorLayout *l=ref->layout;if(l->rank<1||l->rank>4||l->arena_slot>=count||slots[l->arena_slot].generation!=l->generation)return -4;
 uint64_t elements=1;
 for(uint32_t i=0;i<l->rank;i++){if(!l->dims[i]||elements>UINT64_MAX/l->dims[i])return -2;elements*=l->dims[i];}
 return elements>ref->storage->rows*ref->storage->cols?-2:0;
}
static TensorLayout contiguous3(uint64_t b,uint64_t m,uint64_t n,uint32_t slot,uint32_t generation){
 TensorLayout l={{b,m,n,0},{m*n,n,1,0},3,0,slot,generation};return l;
}
static TensorLayout transpose_last2(TensorLayout in){
 uint64_t x=in.dims[in.rank-1];in.dims[in.rank-1]=in.dims[in.rank-2];in.dims[in.rank-2]=x;
 x=in.strides[in.rank-1];in.strides[in.rank-1]=in.strides[in.rank-2];in.strides[in.rank-2]=x;return in;
}
static int batch_matmul(const TensorRef *a,const TensorRef *b,TensorRef *out,const ArenaSlot *slots,uint32_t count){
 if(valid(a,slots,count)||valid(b,slots,count)||valid(out,slots,count))return -1;
 TensorLayout *x=a->layout,*y=b->layout,*z=out->layout;
 if(x->rank!=3||y->rank!=3||z->rank!=3||x->dims[0]!=y->dims[0]||x->dims[0]!=z->dims[0]||
    x->dims[1]!=z->dims[1]||y->dims[2]!=z->dims[2]||x->dims[2]!=y->dims[1])return -1;
 for(uint64_t q=0;q<x->dims[0];q++)for(uint64_t i=0;i<x->dims[1];i++)for(uint64_t j=0;j<y->dims[2];j++){
  float sum=0;for(uint64_t k=0;k<x->dims[2];k++)sum+=a->storage->data[q*x->strides[0]+i*x->strides[1]+k*x->strides[2]]*
    b->storage->data[q*y->strides[0]+k*y->strides[1]+j*y->strides[2]];
  out->storage->data[q*z->strides[0]+i*z->strides[1]+j*z->strides[2]]=sum;
 }
 return 0;
}
static int batch_matmul_backward(TensorRef *a,TensorRef *b,const TensorRef *out,const ArenaSlot *slots,uint32_t count){
 if(valid(a,slots,count)||valid(b,slots,count)||valid(out,slots,count)||!a->storage->grad||!b->storage->grad||!out->storage->grad)return -1;
 TensorLayout *x=a->layout,*y=b->layout,*z=out->layout;
 for(uint64_t q=0;q<x->dims[0];q++)for(uint64_t i=0;i<x->dims[1];i++)for(uint64_t j=0;j<y->dims[2];j++){
  float g=out->storage->grad[q*z->strides[0]+i*z->strides[1]+j*z->strides[2]];
  for(uint64_t k=0;k<x->dims[2];k++){
   uint64_t ai=q*x->strides[0]+i*x->strides[1]+k*x->strides[2],bi=q*y->strides[0]+k*y->strides[1]+j*y->strides[2];
   a->storage->grad[ai]+=g*b->storage->data[bi];b->storage->grad[bi]+=g*a->storage->data[ai];
  }
 }
 return 0;
}
static float value(uint64_t i,uint64_t salt){return (float)((int)((i*43+salt*17)%31)-15)/19.0f;}
static int closef(float a,float b){return fabsf(a-b)<=2e-5f*fmaxf(1.0f,fmaxf(fabsf(a),fabsf(b)));}
static float *mem(uint64_t n){float *p=calloc(n,sizeof(float));if(!p)exit(2);return p;}

static int one(uint64_t m,uint64_t k,uint64_t n,uint64_t *saved){
 const uint64_t batches=2,ac=batches*m*k,kc=batches*n*k,oc=batches*m*n;ArenaSlot slots[3]={{11},{12},{13}};
 TensorView av={mem(ac),mem(ac),batches*m,k},kv={mem(kc),mem(kc),batches*n,k},ov={mem(oc),mem(oc),batches*m,n};
 TensorLayout al=contiguous3(batches,m,k,0,11),kl=contiguous3(batches,n,k,1,12),kt=transpose_last2(kl),ol=contiguous3(batches,m,n,2,13);
 TensorRef a={&av,&al},b={&kv,&kt},o={&ov,&ol};for(uint64_t i=0;i<ac;i++)av.data[i]=value(i,1);for(uint64_t i=0;i<kc;i++)kv.data[i]=value(i,2);for(uint64_t i=0;i<oc;i++)ov.grad[i]=value(i,3);
 if(kt.dims[1]!=k||kt.dims[2]!=n||kt.strides[1]!=1||kt.strides[2]!=k||batch_matmul(&a,&b,&o,slots,3)){fprintf(stderr,"layout/forward failed %llu/%llu/%llu\n",m,k,n);return 1;}
 for(uint64_t q=0;q<batches;q++)for(uint64_t i=0;i<m;i++)for(uint64_t j=0;j<n;j++){float ref=0;for(uint64_t p=0;p<k;p++)ref+=av.data[(q*m+i)*k+p]*kv.data[(q*n+j)*k+p];if(!closef(ref,ov.data[(q*m+i)*n+j])){fprintf(stderr,"reference failed %llu/%llu/%llu\n",m,k,n);return 1;}}
 if(batch_matmul_backward(&a,&b,&o,slots,3)){fprintf(stderr,"backward failed\n");return 1;}
 uint64_t probe=ac/2;float analytic=av.grad[probe],old=av.data[probe],eps=1e-3f,loss[2];
 for(int side=0;side<2;side++){av.data[probe]=old+(side?eps:-eps);batch_matmul(&a,&b,&o,slots,3);loss[side]=0;for(uint64_t i=0;i<oc;i++)loss[side]+=ov.data[i]*ov.grad[i];}av.data[probe]=old;
 float numeric=(loss[1]-loss[0])/(2*eps);
 if(fabsf(analytic-numeric)>2e-3f*fmaxf(1.0f,fmaxf(fabsf(analytic),fabsf(numeric)))){fprintf(stderr,"gradient failed %llu/%llu/%llu analytic=%g numeric=%g\n",m,k,n,analytic,numeric);return 1;}
 slots[1].generation++;if(valid(&b,slots,3)!=-4){fprintf(stderr,"stale failed\n");return 1;}
 *saved+=kc*sizeof(float);free(av.data);free(av.grad);free(kv.data);free(kv.grad);free(ov.data);free(ov.grad);return 0;
}
int main(void){
 const uint64_t dims[]={7,8,9};uint64_t saved=0;unsigned cases=0;
 for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++)for(unsigned k=0;k<3;k++){if(one(dims[i],dims[j],dims[k],&saved))return 1;cases++;}
 printf("tensor layout spike passed: TensorView=32 sidecar=80 TensorRef=16 cases=%u batch=2 dims=7/8/9 transpose=O(1) forward=materialized-reference backward=finite-difference stale=detected avoided_copy_bytes=%llu\n",cases,(unsigned long long)saved);
 return 0;
}
