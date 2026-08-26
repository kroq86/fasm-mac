#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float *data,*grad; uint64_t rows,cols; } Tensor;
typedef struct { uint64_t dims[4],strides[4]; uint32_t rank,flags,arena_slot,generation; } Layout;
typedef struct { Tensor *storage; Layout *layout; } TensorRef;
extern int tensor_matmul_forward_f32(const Tensor*,const Tensor*,Tensor*);
extern int tensor_matmul_backward_f32(Tensor*,Tensor*,const Tensor*);

static float *allocf(uint64_t n){float *p=calloc(n,sizeof(float));if(!p)exit(2);return p;}
static float seed(uint64_t i,uint64_t salt){return (float)((int)((i*31+salt*17)%29)-14)/23.0f;}
static uint64_t offset4(const Layout *l,uint64_t b,uint64_t h,uint64_t t,uint64_t d){return b*l->strides[0]+h*l->strides[1]+t*l->strides[2]+d*l->strides[3];}
static int closef(float a,float b,float tol){return fabsf(a-b)<=tol*fmaxf(1.0f,fmaxf(fabsf(a),fabsf(b)));}

static Layout qkv_slice(uint64_t batch,uint64_t tokens,uint64_t heads,uint64_t dim){
 /* Backing order is [B,T,3,H,D]; the selected Q/K/V base pointer supplies
    the omitted axis, and this view permutes [B,T,H,D] to [B,H,T,D]. */
 Layout l={{batch,heads,tokens,dim},{tokens*3*heads*dim,dim,3*heads*dim,1},4,0,0,1};return l;
}
static int can_flatten_heads_without_copy(const Layout *l){
 /* [B,H,T,D] -> [B,T,H*D] requires H and D to be adjacent after T/H swap. */
 return l->strides[1]==l->dims[3]*l->strides[3]&&l->strides[2]==l->dims[1]*l->dims[3]*l->strides[3];
}
static void merge_heads_forward(const TensorRef *in,float *out){
 Layout *l=in->layout;uint64_t model=l->dims[1]*l->dims[3];
 for(uint64_t b=0;b<l->dims[0];b++)for(uint64_t t=0;t<l->dims[2];t++)for(uint64_t h=0;h<l->dims[1];h++)for(uint64_t d=0;d<l->dims[3];d++)
  out[(b*l->dims[2]+t)*model+h*l->dims[3]+d]=in->storage->data[offset4(l,b,h,t,d)];
}
static void merge_heads_backward(TensorRef *in,const float *out_grad){
 Layout *l=in->layout;uint64_t model=l->dims[1]*l->dims[3];
 for(uint64_t b=0;b<l->dims[0];b++)for(uint64_t t=0;t<l->dims[2];t++)for(uint64_t h=0;h<l->dims[1];h++)for(uint64_t d=0;d<l->dims[3];d++)
  in->storage->grad[offset4(l,b,h,t,d)]+=out_grad[(b*l->dims[2]+t)*model+h*l->dims[3]+d];
}

static int one(uint64_t tokens,uint64_t dim,uint64_t *split_saved,uint64_t *merge_copy){
 const uint64_t batch=2,heads=3,features=7,model=heads*dim,rows=batch*tokens;
 Tensor x={allocf(rows*features),allocf(rows*features),rows,features};
 Tensor w={allocf(features*3*model),allocf(features*3*model),features,3*model};
 Tensor packed={allocf(rows*3*model),allocf(rows*3*model),rows,3*model};
 for(uint64_t i=0;i<rows*features;i++)x.data[i]=seed(i,1);for(uint64_t i=0;i<features*3*model;i++)w.data[i]=seed(i,2)*.2f;
 if(tensor_matmul_forward_f32(&x,&w,&packed))return 1;
 Layout ql=qkv_slice(batch,tokens,heads,dim),kl=ql,vl=ql;
 Tensor qstore=packed,kstore=packed,vstore=packed;qstore.data=packed.data;qstore.grad=packed.grad;kstore.data=packed.data+model;kstore.grad=packed.grad+model;vstore.data=packed.data+2*model;vstore.grad=packed.grad+2*model;
 TensorRef q={&qstore,&ql},k={&kstore,&kl},v={&vstore,&vl};
 for(uint64_t b=0;b<batch;b++)for(uint64_t t=0;t<tokens;t++)for(uint64_t h=0;h<heads;h++)for(uint64_t d=0;d<dim;d++){
  uint64_t packed_row=b*tokens+t,local=h*dim+d;
  if(q.storage->data[offset4(&ql,b,h,t,d)]!=packed.data[packed_row*3*model+local]||
     k.storage->data[offset4(&kl,b,h,t,d)]!=packed.data[packed_row*3*model+model+local]||
     v.storage->data[offset4(&vl,b,h,t,d)]!=packed.data[packed_row*3*model+2*model+local])return 1;
 }
 if(can_flatten_heads_without_copy(&q.layout[0])){fprintf(stderr,"invalid zero-copy merge claim\n");return 1;}
 float *merged=allocf(rows*model),*merged_grad=allocf(rows*model);for(uint64_t i=0;i<rows*model;i++)merged_grad[i]=seed(i,3);
 merge_heads_forward(&q,merged);for(uint64_t i=0;i<rows*model;i++){uint64_t row=i/model,h=(i%model)/dim,d=i%dim;if(merged[i]!=q.storage->data[offset4(&ql,row/tokens,h,row%tokens,d)])return 1;}
 merge_heads_backward(&q,merged_grad);for(uint64_t i=0;i<rows*model;i++){uint64_t row=i/model,h=(i%model)/dim,d=i%dim;if(q.storage->grad[offset4(&ql,row/tokens,h,row%tokens,d)]!=merged_grad[i])return 1;}
 /* The packed Q gradient is a valid output seed for the fused projection. */
 if(tensor_matmul_backward_f32(&x,&w,&packed))return 1;
 uint64_t probe=features*3*model/2;float analytic=w.grad[probe],old=w.data[probe],eps=1e-3f,loss[2];
 for(int side=0;side<2;side++){w.data[probe]=old+(side?eps:-eps);tensor_matmul_forward_f32(&x,&w,&packed);loss[side]=0;for(uint64_t i=0;i<rows*model;i++)loss[side]+=packed.data[(i/model)*3*model+(i%model)]*merged_grad[i];}w.data[probe]=old;
 float numeric=(loss[1]-loss[0])/(2*eps);if(!closef(analytic,numeric,3e-3f))return 1;
 *split_saved+=2*rows*model*sizeof(float);*merge_copy+=rows*model*sizeof(float);
 free(x.data);free(x.grad);free(w.data);free(w.grad);free(packed.data);free(packed.grad);free(merged);free(merged_grad);return 0;
}
int main(void){const uint64_t dims[]={7,8,9};uint64_t split_saved=0,merge_copy=0;for(unsigned t=0;t<3;t++)for(unsigned d=0;d<3;d++)if(one(dims[t],dims[d],&split_saved,&merge_copy))return 1;printf("tensor QKV layout passed: cases=9 packed_projection=[B*T,3*H*D] split_heads=O(1) merge_heads=CONTIGUOUS_ACTION backward=finite-difference split_copy_avoided=%llu merge_copy_required=%llu\n",(unsigned long long)split_saved,(unsigned long long)merge_copy);return 0;}
