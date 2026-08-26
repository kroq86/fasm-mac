#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { OK, IO, FORMAT, SHAPE, CHECKSUM };
enum { F32 = 1, MAX_TENSORS = 8, MAX_RANK = 4, FILE_HEADER = 24, TENSOR_HEADER = 40 };
typedef struct { char name[16]; uint32_t dtype, rank, dims[MAX_RANK], count; float *data; } Parameter;

static void put32(unsigned char *p, uint32_t v) { p[0]=v; p[1]=v>>8; p[2]=v>>16; p[3]=v>>24; }
static uint32_t get32(const unsigned char *p) { return p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24; }
static uint32_t hash(const unsigned char *p,size_t n){uint32_t h=2166136261u;for(size_t i=0;i<n;i++){h^=p[i];h*=16777619u;}return h;}
static int valid(const Parameter*p){uint64_t n=1;if(!p->rank||p->rank>MAX_RANK||p->dtype!=F32)return 0;for(uint32_t i=0;i<p->rank;i++)n*=p->dims[i];return n==p->count;}

static void tensor_header(unsigned char*out,const Parameter*p){
    memset(out,0,TENSOR_HEADER);memcpy(out,p->name,16);put32(out+16,p->dtype);put32(out+20,p->rank);
    for(uint32_t i=0;i<MAX_RANK;i++)put32(out+24+4*i,p->dims[i]);
}

static int save(const char*path,const Parameter*p,uint32_t n){
    if(!n||n>MAX_TENSORS)return FORMAT;uint64_t payload64=0;
    for(uint32_t i=0;i<n;i++){if(!valid(&p[i]))return SHAPE;payload64+=TENSOR_HEADER+(uint64_t)p[i].count*4;}
    if(payload64>UINT32_MAX)return FORMAT;uint32_t payload=(uint32_t)payload64,total=FILE_HEADER+payload;
    unsigned char*wire=malloc(total);if(!wire)return IO;memcpy(wire,"FASMTNSR",8);put32(wire+8,1);put32(wire+12,n);put32(wire+16,payload);
    unsigned char*at=wire+FILE_HEADER;for(uint32_t i=0;i<n;i++){tensor_header(at,&p[i]);at+=TENSOR_HEADER;for(uint32_t j=0;j<p[i].count;j++,at+=4){uint32_t bits;memcpy(&bits,&p[i].data[j],4);put32(at,bits);}}
    put32(wire+20,hash(wire+FILE_HEADER,payload));FILE*f=fopen(path,"wb");int ok=f&&fwrite(wire,1,total,f)==total;if(f&&fclose(f))ok=0;free(wire);return ok?OK:IO;
}

static int load(const char*path,Parameter*p,uint32_t n){
    FILE*f=fopen(path,"rb");if(!f||fseek(f,0,SEEK_END)){if(f)fclose(f);return IO;}long sz=ftell(f);
    if(sz<FILE_HEADER||fseek(f,0,SEEK_SET)){fclose(f);return FORMAT;}unsigned char*w=malloc((size_t)sz);if(!w){fclose(f);return IO;}
    int good=fread(w,1,(size_t)sz,f)==(size_t)sz&&fclose(f)==0;if(!good){free(w);return IO;}
    uint32_t payload=get32(w+16);if(memcmp(w,"FASMTNSR",8)||get32(w+8)!=1||get32(w+12)!=n||payload!=(uint32_t)(sz-FILE_HEADER)){free(w);return FORMAT;}
    if(hash(w+FILE_HEADER,payload)!=get32(w+20)){free(w);return CHECKSUM;}const unsigned char*at=w+FILE_HEADER,*end=at+payload;
    for(uint32_t i=0;i<n;i++){if((size_t)(end-at)<TENSOR_HEADER){free(w);return FORMAT;}uint32_t rank=get32(at+20);
        if(memcmp(at,p[i].name,16)||get32(at+16)!=p[i].dtype||rank!=p[i].rank){free(w);return SHAPE;}uint64_t count=1;
        for(uint32_t d=0;d<MAX_RANK;d++){uint32_t dim=get32(at+24+4*d);if(dim!=p[i].dims[d]){free(w);return SHAPE;}if(d<rank)count*=dim;}at+=TENSOR_HEADER;
        if(count!=p[i].count||count>(uint64_t)(end-at)/4){free(w);return SHAPE;}for(uint32_t j=0;j<p[i].count;j++,at+=4){uint32_t bits=get32(at);memcpy(&p[i].data[j],&bits,4);}}
    int result=at==end?OK:FORMAT;free(w);return result;
}

static int patch(const char*path,long off,unsigned char v,int from_end){FILE*f=fopen(path,"r+b");if(!f||fseek(f,off,from_end?SEEK_END:SEEK_SET)){if(f)fclose(f);return 0;}int ok=fwrite(&v,1,1,f)==1&&fclose(f)==0;return ok;}
static int truncate_copy(const char*src,const char*dst){FILE*i=fopen(src,"rb"),*o=fopen(dst,"wb");if(!i||!o||fseek(i,0,SEEK_END))return 0;long n=ftell(i);if(n<1||fseek(i,0,SEEK_SET))return 0;for(long x=0;x<n-1;x++){int c=fgetc(i);if(c==EOF||fputc(c,o)==EOF)return 0;}return fclose(i)==0&&fclose(o)==0;}

int main(int argc,char**argv){if(argc!=3)return 2;float a[12],b[4],la[12]={0},lb[4]={0};for(int i=0;i<12;i++)a[i]=(i-5)*.125f;for(int i=0;i<4;i++)b[i]=(i+1)*-.25f;
    Parameter src[]={{"wq",F32,2,{3,4},12,a},{"wo",F32,2,{2,2},4,b}},dst[]={{"wq",F32,2,{3,4},12,la},{"wo",F32,2,{2,2},4,lb}};
    if(save(argv[1],src,2)||load(argv[1],dst,2)||memcmp(a,la,sizeof a)||memcmp(b,lb,sizeof b))return 3;
    Parameter wrong[]={{"wq",F32,2,{4,3},12,la},{"wo",F32,2,{2,2},4,lb}};if(load(argv[1],wrong,2)!=SHAPE)return 4;
    if(!truncate_copy(argv[1],argv[2])||load(argv[2],dst,2)!=FORMAT)return 5;if(!patch(argv[1],8,2,0)||load(argv[1],dst,2)!=FORMAT)return 6;
    if(save(argv[1],src,2)||!patch(argv[1],-1,0xff,1)||load(argv[1],dst,2)!=CHECKSUM)return 7;
    puts("tensor checkpoint wire spike passed: version=1 endian=little header=24 tensor_header=40 roundtrip=exact shape/version/truncation/corruption=rejected");return 0;}
