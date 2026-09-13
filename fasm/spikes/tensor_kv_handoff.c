/* Standalone native integration spike; no Python in the execution path. */
#include "tensor_gpt2_handoff.h"
#include "tensor_gpt2_bpe.h"
#include <stdio.h>
#include <stdlib.h>

void gpt2_matmul_bias_accelerate(const float *,int,int,const float *,const float *,int,float *);
static Gpt2Weights sender, receiver;
static HandoffSession source, target;
static HandoffBridge bridge;
static Gpt2Bpe tokenizer;
static float logits[GPT2_VOCAB];
static double worst;

static int checked_path(char *out,size_t cap,const char *dir,const char *name) {
    int n=snprintf(out,cap,"%s/%s",dir,name); return n<0 || (size_t)n>=cap ? -1:0;
}
static int fingerprint(const char *path,const char *sha) {
    char actual[65]; return strlen(sha)!=64 || sha256_file(path,actual) || strcmp(actual,sha);
}
static int compare(SafetensorsFile *ref,const char *name,const float *actual,size_t n,int rows) {
    float *expected=malloc(n*sizeof(float)); if(!expected) return -1;
    uint64_t shape[2]={(uint64_t)(rows?rows:GPT2_VOCAB),768};
    if(st_read_f32(ref,name,shape,rows?2:1,expected)) { free(expected); return -1; }
    for(size_t i=0;i<n;i++) {
        double delta=fabs((double)actual[i]-expected[i]);
        if(!isfinite(actual[i]) || !isfinite(expected[i]) || delta>1e-3+1e-4*fabs(expected[i])) {
            fprintf(stderr,"divergence tensor=%s index=%zu native=%.9g oracle=%.9g delta=%.9g\n",name,i,actual[i],expected[i],delta);
            free(expected); return -1;
        }
        if(delta>worst) worst=delta;
    }
    free(expected); return 0;
}
static int run(const char *sequence,SafetensorsFile *ref,int case_id,int expected) {
    if(strlen(sequence)!=32) return -1;
    char text[100]="Sequence:"; size_t p=strlen(text);
    for(int i=0;i<32;i++) {
        if(sequence[i]!='A' && sequence[i]!='B') return -1;
        text[p++]=' '; text[p++]=sequence[i];
    }
    text[p]=0; int ids[64]; int n=gpt2_bpe_encode(&tokenizer,text,(int)p,ids,64);
    if(n<=0 || n+2>64) return -1;
    if(handoff_init(&source,&sender,12,gpt2_matmul_bias_accelerate) || handoff_init(&target,&receiver,6,gpt2_matmul_bias_accelerate)) return -1;
    for(int i=0;i<n;i++) if(handoff_token(&source,ids[i],NULL)) return -1;
    char name[80];
    if(ref) for(int l=0;l<6;l++) for(int stream=0;stream<2;stream++) {
        snprintf(name,sizeof name,"case%d.source_%c.%d",case_id,stream?'v':'k',l);
        if(compare(ref,name,stream?source.layer[2*l+1].v:source.layer[2*l+1].k,(size_t)n*M,n)) return -1;
    }
    if(handoff_import(&target,&source,&bridge)) return -1;
    /* A second import must reject an occupied receiver, leaving it intact. */
    if(!handoff_import(&target,&source,&bridge)) return -1;
    if(ref) for(int l=0;l<6;l++) for(int stream=0;stream<2;stream++) {
        snprintf(name,sizeof name,"case%d.adapt_%c.%d",case_id,stream?'v':'k',l);
        if(compare(ref,name,stream?target.layer[l].v:target.layer[l].k,(size_t)n*M,n)) return -1;
    }
    int query[16]; int nq=gpt2_bpe_encode(&tokenizer,"Class:",6,query,16);
    if(nq<=0 || n+nq>MAXCACHE) return -1;
    for(int i=0;i<nq;i++) if(handoff_token(&target,query[i],i==nq-1?logits:NULL)) return -1;
    if(ref) {
        snprintf(name,sizeof name,"case%d.logits",case_id);
        if(compare(ref,name,logits,GPT2_VOCAB,0)) return -1;
        float expected_logits[GPT2_VOCAB]; uint64_t shape[1]={GPT2_VOCAB};
        if(st_read_f32(ref,name,shape,1,expected_logits)) return -1;
        int a=0,b=0;
        for(int i=0;i<GPT2_VOCAB;i++) { if(logits[i]>logits[a]) a=i; if(expected_logits[i]>expected_logits[b]) b=i; }
        if(a!=b) { fprintf(stderr,"argmax mismatch: %d vs %d\n",a,b); return -1; }
    }
    int predicted=0;
    for(int i=0;i<GPT2_VOCAB;i++) { if(!isfinite(logits[i])) return -1; if(logits[i]>logits[predicted]) predicted=i; }
    char decoded[100];
    int decoded_len=gpt2_bpe_decode(&tokenizer,&predicted,1,decoded,sizeof decoded-1);
    if(decoded_len<0 || decoded_len>=(int)sizeof decoded) return -1;
    decoded[decoded_len]=0; /* BPE returns raw byte count, not a C string. */
    printf("case=%d predicted_token=%d answer=%s target_token=%d prefix_tokens=%d cache_tokens=%u\n",case_id,predicted,decoded,expected,n,target.layer[0].graph[1].tensor.aux_count);
    fflush(stdout); return predicted==expected?1:0;
}

int main(int argc,char **argv) {
    if(argc!=8 || (strcmp(argv[5],"--sequence") && strcmp(argv[5],"--verify"))) {
        fprintf(stderr,"usage: tensor-kv-handoff ASSET_DIR SENDER_SHA RECEIVER_SHA BRIDGE_SHA --sequence AB... UNUSED | --verify CASES REFERENCE\n"); return 2;
    }
    const char *dir=argv[1]; char path[4096],vocab[4096],merges[4096];
    if(checked_path(path,sizeof path,dir,"sender.safetensors") || gpt2_load_weights_layers(&sender,path,argv[2],12)) return 2;
    if(checked_path(path,sizeof path,dir,"receiver.safetensors") || gpt2_load_weights_layers(&receiver,path,argv[3],6)) return 2;
    if(checked_path(path,sizeof path,dir,"bridge.safetensors") || fingerprint(path,argv[4]) || handoff_load_bridge(&bridge,path)) return 2;
    if(checked_path(vocab,sizeof vocab,dir,"vocab.json") || checked_path(merges,sizeof merges,dir,"merges.txt") || gpt2_bpe_load(&tokenizer,vocab,merges)) return 2;
    if(!strcmp(argv[5],"--sequence")) {
        if(run(argv[6],NULL,0,-1)<0) { fprintf(stderr,"invalid sequence or execution failure\n"); return 2; }
        return 0;
    }
    FILE *cases=fopen(argv[6],"r"); SafetensorsFile refs;
    if(!cases) return 2;
    if(st_open(&refs,argv[7])) { fclose(cases); return 2; }
    char line[128],seq[33],extra; int expected, total=0,correct=0,failed=0;
    while(fgets(line,sizeof line,cases)) {
        if(sscanf(line,"%32s %d %c",seq,&expected,&extra)!=2 || expected<0 || expected>=GPT2_VOCAB || total>=32) { failed=1; break; }
        int r=run(seq,&refs,total,expected); if(r<0) { failed=1; break; }
        total++; correct+=r;
    }
    if(ferror(cases)) failed=1;
    fclose(cases); st_close(&refs);
    if(failed || total!=32) return 3;
    printf("PASS cases=%d correct=%d split=original_dev worst_abs=%.9g atol=0.001 rtol=0.0001\n",total,correct,worst);
    return 0;
}
