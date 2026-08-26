#include "tensor_mlp_executor_spike.h"
#include <stdio.h>
#include <stdlib.h>

enum { PARAMS = IN*HID + HID + HID*OUT + OUT, HEADER = 20 };
static const float data[4][IN]={{0,0},{0,1},{1,0},{1,1}},labels[4]={0,1,1,0};
static void put32(unsigned char*p,uint32_t v){p[0]=v;p[1]=v>>8;p[2]=v>>16;p[3]=v>>24;}
static uint32_t get32(const unsigned char*p){return p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint32_t hash(const unsigned char*p,size_t n){uint32_t h=2166136261u;for(size_t i=0;i<n;i++){h^=p[i];h*=16777619u;}return h;}
static void gather(const Block*b,float*p){memcpy(p,b->w1,sizeof b->w1);p+=IN*HID;memcpy(p,b->b1,sizeof b->b1);p+=HID;memcpy(p,b->w2,sizeof b->w2);p+=HID*OUT;memcpy(p,b->b2,sizeof b->b2);}
static void scatter(Block*b,const float*p){memcpy(b->w1,p,sizeof b->w1);p+=IN*HID;memcpy(b->b1,p,sizeof b->b1);p+=HID;memcpy(b->w2,p,sizeof b->w2);p+=HID*OUT;memcpy(b->b2,p,sizeof b->b2);}
static int checkpoint(const char*path,const Block*b,uint32_t epoch){unsigned char wire[HEADER+PARAMS*4];float p[PARAMS];gather(b,p);memcpy(wire,"MLPCKPT1",8);put32(wire+8,epoch);put32(wire+12,PARAMS);
    for(unsigned i=0;i<PARAMS;i++){uint32_t bits;memcpy(&bits,&p[i],4);put32(wire+HEADER+4*i,bits);}put32(wire+16,hash(wire+HEADER,PARAMS*4));FILE*f=fopen(path,"wb");int ok=f&&fwrite(wire,1,sizeof wire,f)==sizeof wire;if(f&&fclose(f))ok=0;return ok?0:-1;}
static int restore(const char*path,Block*b,uint32_t*epoch){unsigned char wire[HEADER+PARAMS*4];FILE*f=fopen(path,"rb");if(!f)return-1;int ok=fread(wire,1,sizeof wire,f)==sizeof wire&&fgetc(f)==EOF&&fclose(f)==0;
    if(!ok||memcmp(wire,"MLPCKPT1",8)||get32(wire+12)!=PARAMS||get32(wire+16)!=hash(wire+HEADER,PARAMS*4))return-1;float p[PARAMS];for(unsigned i=0;i<PARAMS;i++){uint32_t bits=get32(wire+HEADER+4*i);memcpy(&p[i],&bits,4);}scatter(b,p);*epoch=get32(wire+8);return 0;}
static int train(Block*b,unsigned begin,unsigned end){Scratch scratch={0};ExecStep steps[8];Context ctx[8];uint32_t generation=700;
    for(unsigned epoch=begin;epoch<end;epoch++)for(int sample=0;sample<4;sample++){memcpy(b->x,data[sample],sizeof b->x);mlp_forward(b);float seed[OUT]={b->out[0]-labels[sample]};unsigned n=mlp_emit(b,&scratch,seed,&generation,steps,ctx);if(n!=8||tensor_transformer_steps_execute(steps,n))return-1;
        for(int i=0;i<IN*HID;i++)b->w1[i]-=.1f*b->dw1[i];for(int i=0;i<HID;i++)b->b1[i]-=.1f*b->db1[i];for(int i=0;i<HID*OUT;i++)b->w2[i]-=.1f*b->dw2[i];b->b2[0]-=.1f*b->db2[0];}return 0;}
static void report(Block*b){float loss=0,pred[4];for(int i=0;i<4;i++){memcpy(b->x,data[i],sizeof b->x);mlp_forward(b);pred[i]=b->out[0];float e=pred[i]-labels[i];loss+=e*e;}printf("epoch=4000 loss=%.9g predictions=%.9g,%.9g,%.9g,%.9g\n",loss/4,pred[0],pred[1],pred[2],pred[3]);}
int main(int argc,char**argv){if(argc<3)return 2;Block b;mlp_initialize(&b);
    if(!strcmp(argv[1],"full")){if(argc!=3||train(&b,0,4000)||checkpoint(argv[2],&b,4000))return 3;report(&b);return 0;}
    if(!strcmp(argv[1],"split")){if(argc!=3||train(&b,0,2000)||checkpoint(argv[2],&b,2000))return 4;return 0;}
    if(!strcmp(argv[1],"resume")){uint32_t epoch;if(argc!=4||restore(argv[2],&b,&epoch)||epoch!=2000||train(&b,epoch,4000)||checkpoint(argv[3],&b,4000))return 5;report(&b);return 0;}return 2;}
