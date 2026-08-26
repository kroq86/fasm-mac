#define TRANSFORMER_EXECUTOR_NO_MAIN
#include "tensor_transformer_executor_spike.h"
#undef TRANSFORMER_EXECUTOR_NO_MAIN
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct{char magic[8];uint32_t rows,tokens,features,reserved;}Header;
typedef struct{uint32_t test,id;float x[T*M],y;}Record;
static int shuffle_bands;
static int exact(void*p,size_t n,FILE*f){return fread(p,1,n,f)==n?0:-1;}static uint32_t rng=0x47554e31u;static uint32_t random_u32(void){rng=rng*1664525u+1013904223u;return rng;}
static void load_input(Block*b,const Record*r){int order[T]={0,1,2};if(shuffle_bands){uint32_t z=r->id*747796405u+2891336453u;for(int i=T-1;i;i--){z=z*1664525u+1013904223u;int j=z%(i+1),q=order[i];order[i]=order[j];order[j]=q;}}for(int t=0;t<T;t++)for(int j=0;j<M;j++)b->x[t*M+j]=r->x[order[t]*M+j];}
static float sigmoidf_(float z){if(z>60.f)z=60.f;if(z<-60.f)z=-60.f;return 1.f/(1.f+expf(-z));}
static float prediction(const Block*b,const float*w,float bias){float p=bias;for(int t=0;t<T;t++)for(int j=0;j<M;j++)p+=b->out[t*M+j]*w[j]/T;return sigmoidf_(p);}
static float evaluate(Block*b,const Record*r,const uint32_t*ids,uint32_t count,const float*w,float bias){uint32_t correct=0;for(uint32_t q=0;q<count;q++){load_input(b,&r[ids[q]]);forward(b);float p=prediction(b,w,bias);if((p>=.5f)==(r[ids[q]].y>=.5f))correct++;}return(float)correct/count;}
int main(int argc,char**argv){if(argc!=3||(strcmp(argv[2],"ordered")&&strcmp(argv[2],"shuffled"))){fprintf(stderr,"usage: %s sequence.bin ordered|shuffled\n",argv[0]);return 2;}shuffle_bands=!strcmp(argv[2],"shuffled");FILE*f=fopen(argv[1],"rb");if(!f)return 2;Header h;if(exact(&h,sizeof h,f)||memcmp(h.magic,"GUNSEQ1",7)||h.tokens!=T||h.features!=M)return 2;Record*r=calloc(h.rows,sizeof*r);if(!r)return 2;for(uint32_t i=0;i<h.rows;i++){if(exact(&r[i].test,4,f)||exact(&r[i].id,4,f)||exact(r[i].x,sizeof r[i].x,f)||exact(&r[i].y,4,f))return 2;}fclose(f);uint32_t*train=malloc(h.rows*4),*test=malloc(h.rows*4),nt=0,nv=0;if(!train||!test)return 2;for(uint32_t i=0;i<h.rows;i++)(r[i].test?test:train)[r[i].test?nv++:nt++]=i;Block b;float unused[T*M];initialize(&b,unused);float head[M];for(int j=0;j<M;j++)head[j]=initv(j,19)*.2f;float bias=0.f;Scratch scratch={0};ExecStep steps[21];Context ctx[21];uint32_t generation=41;float initial=evaluate(&b,r,test,nv,head,bias);enum{EPOCHS=2500};for(unsigned epoch=0;epoch<EPOCHS;epoch++){for(uint32_t i=nt-1;i;i--){uint32_t j=random_u32()%(i+1),x=train[i];train[i]=train[j];train[j]=x;}for(uint32_t q=0;q<nt;q++){Record*sample=&r[train[q]];load_input(&b,sample);forward(&b);float p=prediction(&b,head,bias),error=p-sample->y,seed[T*M],head_grad[M]={0};for(int j=0;j<M;j++){float pooled=0;for(int t=0;t<T;t++)pooled+=b.out[t*M+j]/T;head_grad[j]=error*pooled;for(int t=0;t<T;t++)seed[t*M+j]=error*head[j]/T;}unsigned count=emit(&b,&scratch,seed,&generation,steps,ctx);if(count!=21||tensor_transformer_steps_execute(steps,count))return 3;float lr=.01f;for(int i=0;i<M*QW;i++)b.wq[i]-=lr*b.dwq[i];for(int i=0;i<M*M;i++)b.wo[i]-=lr*b.dwo[i];for(int i=0;i<M*F;i++)b.w1[i]-=lr*b.dw1[i];for(int i=0;i<F*M;i++)b.w2[i]-=lr*b.dw2[i];for(int j=0;j<M;j++)head[j]-=.02f*head_grad[j];bias-=.02f*error;}}
 float final=evaluate(&b,r,test,nv,head,bias);printf("gunpoint transformer trained: mode=%s samples=%u train=%u test=%u tokens=%d features=%d epochs=%u backward_actions=21 test_acc=%.3f->%.3f\n",argv[2],h.rows,nt,nv,T,M,(unsigned)EPOCHS,initial,final);free(r);free(train);free(test);return isfinite(final)?0:1;}
