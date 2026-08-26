#ifndef TENSOR_CHECKPOINT_V1_SPIKE_H
#define TENSOR_CHECKPOINT_V1_SPIKE_H
/* Experimental checkpoint contract. Not a stable core ABI. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { TC_OK, TC_FORMAT, TC_LIMIT, TC_INTEGRITY, TC_MODEL, TC_TENSOR };
enum { TC_F32=1, TC_OPTIONAL=1, TC_HEADER=64, TC_RECORD=64, TC_MAX_RANK=4, TC_MAX_TENSORS=1024 };
typedef struct { char name[16]; uint32_t dtype,rank,dims[4],flags; const void*data; uint64_t bytes; } TcTensor;
typedef struct { uint64_t graph_fingerprint,epoch; uint32_t tensor_count; } TcInfo;
typedef struct { const char*name;uint32_t dtype,rank,dims[4],flags;void*data;uint64_t bytes; } TcRequest;
static void tc_put32(unsigned char*p,uint32_t v){p[0]=v;p[1]=v>>8;p[2]=v>>16;p[3]=v>>24;}
static uint32_t tc_get32(const unsigned char*p){return p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static void tc_put64(unsigned char*p,uint64_t v){for(unsigned i=0;i<8;i++)p[i]=v>>(8*i);}
static uint64_t tc_get64(const unsigned char*p){uint64_t v=0;for(unsigned i=0;i<8;i++)v|=(uint64_t)p[i]<<(8*i);return v;}
static uint32_t tc_hash(const unsigned char*p,uint64_t n){uint32_t h=2166136261u;for(uint64_t i=0;i<n;i++){h^=p[i];h*=16777619u;}return h;}
static int tc_name(const char n[16]){return n[0]&&memchr(n,0,16)!=NULL;}
static int tc_shape_bytes(uint32_t dtype,uint32_t rank,const uint32_t*dims,uint64_t*bytes){if(dtype!=TC_F32||!rank||rank>4)return 0;uint64_t n=1;for(uint32_t i=0;i<rank;i++){if(!dims[i]||n>UINT64_MAX/dims[i])return 0;n*=dims[i];}if(n>UINT64_MAX/4)return 0;*bytes=n*4;return 1;}

static int tc_encode(const TcTensor*t,uint32_t n,uint64_t fp,uint64_t epoch,unsigned char**out,uint64_t*out_bytes){
    if(!n||n>TC_MAX_TENSORS)return TC_LIMIT;uint64_t payload=0;
    for(uint32_t i=0;i<n;i++){uint64_t shaped;if(!tc_name(t[i].name)||!tc_shape_bytes(t[i].dtype,t[i].rank,t[i].dims,&shaped)||shaped!=t[i].bytes)return TC_FORMAT;for(uint32_t j=i+1;j<n;j++)if(!strncmp(t[i].name,t[j].name,16))return TC_FORMAT;if(UINT64_MAX-payload<t[i].bytes)return TC_LIMIT;payload+=t[i].bytes;}
    uint64_t table=(uint64_t)n*TC_RECORD,total=TC_HEADER+table+payload;if(total>SIZE_MAX)return TC_LIMIT;unsigned char*w=calloc(1,(size_t)total);if(!w)return TC_LIMIT;
    memcpy(w,"FASMCKP1",8);tc_put32(w+8,1);tc_put32(w+12,n);tc_put64(w+16,fp);tc_put64(w+24,epoch);tc_put64(w+32,table);tc_put64(w+40,payload);
    uint64_t off=0;for(uint32_t i=0;i<n;i++){unsigned char*r=w+TC_HEADER+(uint64_t)i*TC_RECORD;memcpy(r,t[i].name,16);tc_put32(r+16,t[i].dtype);tc_put32(r+20,t[i].rank);for(unsigned d=0;d<4;d++)tc_put32(r+24+4*d,t[i].dims[d]);tc_put64(r+40,off);tc_put64(r+48,t[i].bytes);tc_put32(r+56,tc_hash(t[i].data,t[i].bytes));tc_put32(r+60,t[i].flags);memcpy(w+TC_HEADER+table+off,t[i].data,(size_t)t[i].bytes);off+=t[i].bytes;}
    tc_put32(w+48,tc_hash(w+TC_HEADER,table));tc_put32(w+52,tc_hash(w+TC_HEADER+table,payload));tc_put32(w+56,tc_hash(w,56));*out=w;*out_bytes=total;return TC_OK;
}

static int tc_inspect(const unsigned char*w,uint64_t bytes,uint64_t expected_fp,TcInfo*info){
    if(bytes<TC_HEADER||memcmp(w,"FASMCKP1",8)||tc_get32(w+8)!=1||tc_hash(w,56)!=tc_get32(w+56))return TC_FORMAT;uint32_t n=tc_get32(w+12);uint64_t table=tc_get64(w+32),payload=tc_get64(w+40);
    if(!n||n>TC_MAX_TENSORS||table!=(uint64_t)n*TC_RECORD||table>bytes-TC_HEADER||payload!=bytes-TC_HEADER-table)return TC_LIMIT;if(tc_hash(w+TC_HEADER,table)!=tc_get32(w+48)||tc_hash(w+TC_HEADER+table,payload)!=tc_get32(w+52))return TC_INTEGRITY;if(expected_fp&&tc_get64(w+16)!=expected_fp)return TC_MODEL;
    for(uint32_t i=0;i<n;i++){const unsigned char*r=w+TC_HEADER+(uint64_t)i*TC_RECORD;uint32_t dims[4];for(unsigned d=0;d<4;d++)dims[d]=tc_get32(r+24+4*d);uint64_t shaped;if(!tc_name((const char*)r)||!tc_shape_bytes(tc_get32(r+16),tc_get32(r+20),dims,&shaped))return TC_FORMAT;for(uint32_t j=i+1;j<n;j++)if(!memcmp(r,w+TC_HEADER+(uint64_t)j*TC_RECORD,16))return TC_FORMAT;uint64_t off=tc_get64(r+40),len=tc_get64(r+48);if(shaped!=len||off>payload||len>payload-off)return TC_LIMIT;if(tc_hash(w+TC_HEADER+table+off,len)!=tc_get32(r+56))return TC_INTEGRITY;for(uint32_t j=0;j<i;j++){const unsigned char*q=w+TC_HEADER+(uint64_t)j*TC_RECORD;uint64_t qo=tc_get64(q+40),ql=tc_get64(q+48);if(off<qo+ql&&qo<off+len)return TC_LIMIT;}}
    if(info)*info=(TcInfo){tc_get64(w+16),tc_get64(w+24),n};return TC_OK;
}

static int tc_load(const unsigned char*w,uint64_t bytes,uint64_t fp,TcRequest*q,uint32_t wanted){TcInfo info;int rc=tc_inspect(w,bytes,fp,&info);if(rc)return rc;uint64_t table=(uint64_t)info.tensor_count*TC_RECORD;
    /* preflight every request before modifying any destination */
    for(uint32_t x=0;x<wanted;x++){const unsigned char*match=NULL;for(uint32_t i=0;i<info.tensor_count;i++){const unsigned char*r=w+TC_HEADER+(uint64_t)i*TC_RECORD;if(!strncmp((const char*)r,q[x].name,16))match=r;}if(!match){if(q[x].flags&TC_OPTIONAL)continue;return TC_TENSOR;}if(tc_get32(match+16)!=q[x].dtype||tc_get32(match+20)!=q[x].rank||tc_get64(match+48)!=q[x].bytes)return TC_TENSOR;for(unsigned d=0;d<4;d++)if(tc_get32(match+24+4*d)!=q[x].dims[d])return TC_TENSOR;}
    for(uint32_t x=0;x<wanted;x++)for(uint32_t i=0;i<info.tensor_count;i++){const unsigned char*r=w+TC_HEADER+(uint64_t)i*TC_RECORD;if(!strncmp((const char*)r,q[x].name,16))memcpy(q[x].data,w+TC_HEADER+table+tc_get64(r+40),(size_t)q[x].bytes);}
    return TC_OK;}
#endif
