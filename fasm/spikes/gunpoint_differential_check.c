/* First-divergence harness: old model-specific GunPoint oracle versus the
 * canonical semantic graph, on one fixed exported sample. */
#define Block OracleBlock
#define ExecStep OracleExecStep
#define Context OracleContext
#define Scratch OracleScratch
#define initv oracle_initv
#define mm oracle_mm
#define mmback oracle_mmback
#define ln oracle_ln
#define lnback oracle_lnback
#define forward oracle_forward
#define loss oracle_loss
#define backward oracle_backward
#define check oracle_check
#define initialize oracle_initialize
#define emit oracle_emit
#define execute oracle_execute
#define zero_one oracle_zero_one
#define reverse_one oracle_reverse_one
#define tensor_transformer_steps_execute oracle_steps_execute
#define TRANSFORMER_EXECUTOR_NO_MAIN
#include "tensor_transformer_executor_spike.h"
#undef TRANSFORMER_EXECUTOR_NO_MAIN
#undef tensor_transformer_steps_execute
#undef check
#undef reverse_one
#undef zero_one
#undef execute
#undef emit
#undef initialize
#undef backward
#undef loss
#undef forward
#undef lnback
#undef ln
#undef mmback
#undef mm
#undef initv
#undef Scratch
#undef Context
#undef ExecStep
#undef Block

int oracle_steps_execute(const OracleExecStep *steps, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) { int rc = steps[i].run(steps[i].context); if (rc) return rc; }
    return 0;
}

#define main gunpoint_semantic_train_unused_main
#include "gunpoint_semantic_train.c"
#undef main

typedef struct {
    float input[T*M], qkv[T*QW], prob[H*T*T], attn[H*T*D], merged[T*M];
    float proj[T*M], sum1[T*M], ln1[T*M], ff[T*M], sum2[T*M], out[T*M], pooled[M];
    float logit, probability, loss;
    float dwq[M*QW], dwo[M*M], dw1[M*F], dw2[F*M], dhead[M], dbias;
    float next_wq[M*QW], next_wo[M*M], next_w1[M*F], next_w2[F*M], next_head[M], next_bias;
} Snapshot;

static void oracle_snapshot(const Record *r, Snapshot *s) {
    OracleBlock b; float seed[T*M]; oracle_initialize(&b, seed);
    float head[M]; for (int j=0;j<M;j++) head[j]=oracle_initv(j,19)*.2f;
    memcpy(b.x,r->x,sizeof b.x); oracle_forward(&b);
    float pooled[M]={0}, logit=0;
    for(int j=0;j<M;j++){for(int t=0;t<T;t++)pooled[j]+=b.out[t*M+j]/T;logit+=pooled[j]*head[j];}
    float p=1.f/(1.f+expf(-logit)), error=p-r->y;
    for(int j=0;j<M;j++)for(int t=0;t<T;t++)seed[t*M+j]=error*head[j]/T;
    OracleScratch scratch={0}; OracleExecStep steps[21]; OracleContext ctx[21]; uint32_t generation=41;
    unsigned count=oracle_emit(&b,&scratch,seed,&generation,steps,ctx); if(count!=21||oracle_steps_execute(steps,count))abort();
    memcpy(s->input,b.x,sizeof s->input);memcpy(s->qkv,b.qkv,sizeof s->qkv);memcpy(s->prob,b.prob,sizeof s->prob);
    memcpy(s->attn,b.head,sizeof s->attn);memcpy(s->merged,b.merge,sizeof s->merged);memcpy(s->proj,b.proj,sizeof s->proj);
    memcpy(s->sum1,b.s1,sizeof s->sum1);memcpy(s->ln1,b.ln1,sizeof s->ln1);memcpy(s->ff,b.ff,sizeof s->ff);
    memcpy(s->sum2,b.s2,sizeof s->sum2);memcpy(s->out,b.out,sizeof s->out);memcpy(s->pooled,pooled,sizeof s->pooled);
    s->logit=logit;s->probability=p;s->loss=-(r->y*logf(p)+(1-r->y)*logf(1-p));
    memcpy(s->dwq,b.dwq,sizeof s->dwq);memcpy(s->dwo,b.dwo,sizeof s->dwo);memcpy(s->dw1,b.dw1,sizeof s->dw1);memcpy(s->dw2,b.dw2,sizeof s->dw2);
    for(int j=0;j<M;j++)s->dhead[j]=error*pooled[j];s->dbias=error;
    for(int i=0;i<M*QW;i++)s->next_wq[i]=b.wq[i]-.01f*b.dwq[i];for(int i=0;i<M*M;i++)s->next_wo[i]=b.wo[i]-.01f*b.dwo[i];
    for(int i=0;i<M*F;i++)s->next_w1[i]=b.w1[i]-.01f*b.dw1[i];for(int i=0;i<F*M;i++)s->next_w2[i]=b.w2[i]-.01f*b.dw2[i];
    for(int j=0;j<M;j++)s->next_head[j]=head[j]-.02f*s->dhead[j];s->next_bias=-.02f*error;
}

static void canonical_snapshot(const Record *r, Snapshot *s) {
    Model m; if(model_init(&m))abort(); load_input(&m,r,0);
    if(tensor_transformer_steps_execute(m.steps,m.step_count-6))abort();
    memcpy(s->input,m.x,sizeof s->input);memcpy(s->qkv,m.qkv,sizeof s->qkv);memcpy(s->prob,m.attn_aux,sizeof s->prob);
    memcpy(s->attn,m.attn,sizeof s->attn);memcpy(s->merged,m.merged,sizeof s->merged);memcpy(s->proj,m.proj,sizeof s->proj);
    memcpy(s->sum1,m.sum1,sizeof s->sum1);memcpy(s->ln1,m.ln1,sizeof s->ln1);memcpy(s->ff,m.ff,sizeof s->ff);
    memcpy(s->sum2,m.sum2,sizeof s->sum2);memcpy(s->out,m.ln2,sizeof s->out);memcpy(s->pooled,m.mean,sizeof s->pooled);
    s->logit=m.affine[0];s->probability=m.pred[0];s->loss=m.loss[0];
    memcpy(s->dwq,m.gwq,sizeof s->dwq);memcpy(s->dwo,m.gwo,sizeof s->dwo);memcpy(s->dw1,m.gw1,sizeof s->dw1);memcpy(s->dw2,m.gw2,sizeof s->dw2);
    memcpy(s->dhead,m.ghead,sizeof s->dhead);s->dbias=m.gbias[0];
    if(tensor_transformer_steps_execute(m.steps+m.step_count-6,6))abort();
    memcpy(s->next_wq,m.wq,sizeof s->next_wq);memcpy(s->next_wo,m.wo,sizeof s->next_wo);memcpy(s->next_w1,m.w1,sizeof s->next_w1);
    memcpy(s->next_w2,m.w2,sizeof s->next_w2);memcpy(s->next_head,m.head,sizeof s->next_head);s->next_bias=m.bias[0];
}

typedef struct { OracleBlock block; OracleScratch scratch; float head[M], bias, seed[T*M]; uint32_t generation; } OracleTrainingState;
static void oracle_training_init(OracleTrainingState *s){memset(s,0,sizeof *s);oracle_initialize(&s->block,s->seed);for(int j=0;j<M;j++)s->head[j]=oracle_initv(j,19)*.2f;s->generation=41;}
static void oracle_training_step(OracleTrainingState*s,const Record*r,Snapshot*out){
    OracleBlock*b=&s->block;memcpy(b->x,r->x,sizeof b->x);oracle_forward(b);float pooled[M]={0},logit=s->bias;
    for(int j=0;j<M;j++){for(int t=0;t<T;t++)pooled[j]+=b->out[t*M+j]/T;logit+=pooled[j]*s->head[j];}
    float p=1.f/(1.f+expf(-logit)),error=p-r->y,head_grad[M];
    for(int j=0;j<M;j++){head_grad[j]=error*pooled[j];for(int t=0;t<T;t++)s->seed[t*M+j]=error*s->head[j]/T;}
    OracleExecStep steps[21];OracleContext ctx[21];unsigned n=oracle_emit(b,&s->scratch,s->seed,&s->generation,steps,ctx);if(n!=21||oracle_steps_execute(steps,n))abort();
    memcpy(out->input,b->x,sizeof out->input);memcpy(out->qkv,b->qkv,sizeof out->qkv);memcpy(out->prob,b->prob,sizeof out->prob);memcpy(out->attn,b->head,sizeof out->attn);memcpy(out->merged,b->merge,sizeof out->merged);memcpy(out->proj,b->proj,sizeof out->proj);memcpy(out->sum1,b->s1,sizeof out->sum1);memcpy(out->ln1,b->ln1,sizeof out->ln1);memcpy(out->ff,b->ff,sizeof out->ff);memcpy(out->sum2,b->s2,sizeof out->sum2);memcpy(out->out,b->out,sizeof out->out);memcpy(out->pooled,pooled,sizeof out->pooled);
    out->logit=logit;out->probability=p;out->loss=-(r->y*logf(p)+(1-r->y)*logf(1-p));
    memcpy(out->dwq,b->dwq,sizeof out->dwq);memcpy(out->dwo,b->dwo,sizeof out->dwo);memcpy(out->dw1,b->dw1,sizeof out->dw1);memcpy(out->dw2,b->dw2,sizeof out->dw2);memcpy(out->dhead,head_grad,sizeof out->dhead);out->dbias=error;
    for(int i=0;i<M*QW;i++)b->wq[i]-=.01f*b->dwq[i];for(int i=0;i<M*M;i++)b->wo[i]-=.01f*b->dwo[i];for(int i=0;i<M*F;i++)b->w1[i]-=.01f*b->dw1[i];for(int i=0;i<F*M;i++)b->w2[i]-=.01f*b->dw2[i];for(int j=0;j<M;j++)s->head[j]-=.02f*head_grad[j];s->bias-=.02f*error;
    memcpy(out->next_wq,b->wq,sizeof out->next_wq);memcpy(out->next_wo,b->wo,sizeof out->next_wo);memcpy(out->next_w1,b->w1,sizeof out->next_w1);memcpy(out->next_w2,b->w2,sizeof out->next_w2);memcpy(out->next_head,s->head,sizeof out->next_head);out->next_bias=s->bias;
}
static void canonical_training_step(Model*m,const Record*r,Snapshot*out){
    load_input(m,r,0);if(tensor_transformer_steps_execute(m->steps,m->step_count-6))abort();
    out->logit=m->affine[0];out->probability=m->pred[0];out->loss=m->loss[0];memcpy(out->dwq,m->gwq,sizeof out->dwq);memcpy(out->dwo,m->gwo,sizeof out->dwo);memcpy(out->dw1,m->gw1,sizeof out->dw1);memcpy(out->dw2,m->gw2,sizeof out->dw2);memcpy(out->dhead,m->ghead,sizeof out->dhead);out->dbias=m->gbias[0];
    if(tensor_transformer_steps_execute(m->steps+m->step_count-6,6))abort();memcpy(out->next_wq,m->wq,sizeof out->next_wq);memcpy(out->next_wo,m->wo,sizeof out->next_wo);memcpy(out->next_w1,m->w1,sizeof out->next_w1);memcpy(out->next_w2,m->w2,sizeof out->next_w2);memcpy(out->next_head,m->head,sizeof out->next_head);out->next_bias=m->bias[0];
}

/* Diagnostic only: keep the canonical forward graph, but seed affine's
 * gradient with the numerically fused sigmoid+BCE derivative. */
static void canonical_training_step_fused(Model*m,const Record*r,int shuffled,Snapshot*out){
    load_input(m,r,shuffled);
    if(tensor_transformer_steps_execute(m->steps,m->forward_count))abort();
    memcpy(out->input,m->x,sizeof out->input);memcpy(out->qkv,m->qkv,sizeof out->qkv);memcpy(out->prob,m->attn_aux,sizeof out->prob);memcpy(out->attn,m->attn,sizeof out->attn);memcpy(out->merged,m->merged,sizeof out->merged);memcpy(out->proj,m->proj,sizeof out->proj);memcpy(out->sum1,m->sum1,sizeof out->sum1);memcpy(out->ln1,m->ln1,sizeof out->ln1);memcpy(out->ff,m->ff,sizeof out->ff);memcpy(out->sum2,m->sum2,sizeof out->sum2);memcpy(out->out,m->ln2,sizeof out->out);memcpy(out->pooled,m->mean,sizeof out->pooled);
    out->logit=m->affine[0];out->probability=m->pred[0];out->loss=m->loss[0];
    unsigned i=m->forward_count;
    while(i<m->step_count&&m->ctx[i].action==ZERO_GRAD){if(m->steps[i].run(m->steps[i].context))abort();i++;}
    m->gaffine[0]=m->pred[0]-m->target[0];
    while(i<m->step_count&&m->ctx[i].action==BACKWARD){
        if(m->ctx[i].node<=N_AFFINE&&m->steps[i].run(m->steps[i].context))abort();
        i++;
    }
    memcpy(out->dwq,m->gwq,sizeof out->dwq);memcpy(out->dwo,m->gwo,sizeof out->dwo);memcpy(out->dw1,m->gw1,sizeof out->dw1);memcpy(out->dw2,m->gw2,sizeof out->dw2);memcpy(out->dhead,m->ghead,sizeof out->dhead);out->dbias=m->gbias[0];
    while(i<m->step_count){if(m->steps[i].run(m->steps[i].context))abort();i++;}
    memcpy(out->next_wq,m->wq,sizeof out->next_wq);memcpy(out->next_wo,m->wo,sizeof out->next_wo);memcpy(out->next_w1,m->w1,sizeof out->next_w1);memcpy(out->next_w2,m->w2,sizeof out->next_w2);memcpy(out->next_head,m->head,sizeof out->next_head);out->next_bias=m->bias[0];
}

static uint32_t ordered_float_bits(float x){uint32_t u;memcpy(&u,&x,4);return(u&0x80000000u)?~u:u|0x80000000u;}
static uint32_t ulp_distance(float a,float b){uint32_t x=ordered_float_bits(a),y=ordered_float_bits(b);return x>y?x-y:y-x;}
static int compare_at(unsigned epoch,unsigned sample,const char*name,const float*a,const float*b,size_t n);

static int first_bits_reported;
static void note_first_bits(unsigned epoch,unsigned sample,const char*name,const float*a,const float*b,size_t n){if(first_bits_reported)return;for(size_t i=0;i<n;i++)if(memcmp(&a[i],&b[i],sizeof(float))){printf("FIRST_NUMERICAL_DIVERGENCE epoch=%u sample=%u phase=pre_forward tensor=%s index=%zu oracle=%.9g canonical=%.9g delta=%.9g ulp=%u\n",epoch,sample,name,i,a[i],b[i],b[i]-a[i],ulp_distance(a[i],b[i]));first_bits_reported=1;return;}}
static int compare_state(unsigned epoch,unsigned sample,OracleTrainingState*o,Model*c){
#define STATE(name,oa,ca,n) do{note_first_bits(epoch,sample,name,oa,ca,n);if(compare_at(epoch,sample,"pre_forward:" name,oa,ca,n))return 1;}while(0)
    STATE("wq",o->block.wq,c->wq,M*QW);STATE("wo",o->block.wo,c->wo,M*M);STATE("w1",o->block.w1,c->w1,M*F);STATE("w2",o->block.w2,c->w2,F*M);STATE("head",o->head,c->head,M);STATE("bias",&o->bias,c->bias,1);
#undef STATE
    return 0;
}
static void state_max_report(unsigned epoch,unsigned sample,OracleTrainingState*o,Model*c){const char*best_name="none";size_t best_index=0;float best_a=0,best_b=0,best_delta=0;uint32_t best_ulp=0;
#define MAX_STATE(name,oa,ca,n) do{for(size_t k=0;k<(n);k++){float d=fabsf((ca)[k]-(oa)[k]);uint32_t u=ulp_distance((oa)[k],(ca)[k]);if(d>best_delta||(d==best_delta&&u>best_ulp)){best_name=name;best_index=k;best_a=(oa)[k];best_b=(ca)[k];best_delta=d;best_ulp=u;}}}while(0)
    MAX_STATE("wq",o->block.wq,c->wq,M*QW);MAX_STATE("wo",o->block.wo,c->wo,M*M);MAX_STATE("w1",o->block.w1,c->w1,M*F);MAX_STATE("w2",o->block.w2,c->w2,F*M);MAX_STATE("head",o->head,c->head,M);MAX_STATE("bias",&o->bias,c->bias,1);
#undef MAX_STATE
    printf("PRE_FORWARD_STATE_AT_MATERIAL_BOUNDARY epoch=%u sample=%u tensor=%s index=%zu oracle=%.9g canonical=%.9g abs_delta=%.9g ulp=%u optimizer_state=none_sgd\n",epoch,sample,best_name,best_index,best_a,best_b,best_delta,best_ulp);
}
static void state_max_values(OracleTrainingState*o,Model*c,float*max_abs,uint32_t*max_ulp){*max_abs=0;*max_ulp=0;
#define STATE_VALUES(oa,ca,n) do{for(size_t k=0;k<(n);k++){float d=fabsf((ca)[k]-(oa)[k]);uint32_t u=ulp_distance((oa)[k],(ca)[k]);if(d>*max_abs)*max_abs=d;if(u>*max_ulp)*max_ulp=u;}}while(0)
    STATE_VALUES(o->block.wq,c->wq,M*QW);STATE_VALUES(o->block.wo,c->wo,M*M);STATE_VALUES(o->block.w1,c->w1,M*F);STATE_VALUES(o->block.w2,c->w2,F*M);STATE_VALUES(o->head,c->head,M);STATE_VALUES(&o->bias,c->bias,1);
#undef STATE_VALUES
}

static int compare_at(unsigned epoch,unsigned sample,const char*name,const float*a,const float*b,size_t n){for(size_t i=0;i<n;i++){float tol=3e-5f*fmaxf(1.f,fmaxf(fabsf(a[i]),fabsf(b[i])));if(fabsf(a[i]-b[i])>tol){printf("FIRST_DIVERGENCE epoch=%u sample=%u tensor=%s index=%zu oracle=%.9g canonical=%.9g delta=%.9g tolerance=%.9g\n",epoch,sample,name,i,a[i],b[i],b[i]-a[i],tol);return 1;}}return 0;}
#define CMP_AT(field) do{if(compare_at(epoch,position,#field,a.field,b.field,sizeof a.field/sizeof(float)))return 1;}while(0)
static int load_records(const char*path,Record**records,uint32_t**train_out,uint32_t*nt_out,uint32_t**test_out,uint32_t*nv_out){FILE*f=fopen(path,"rb");if(!f)return 2;Header h;if(exact(&h,sizeof h,f))return 2;Record*r=calloc(h.rows,sizeof*r);if(!r)return 2;for(uint32_t i=0;i<h.rows;i++)if(exact(&r[i].test,4,f)||exact(&r[i].id,4,f)||exact(r[i].x,sizeof r[i].x,f)||exact(&r[i].y,4,f))return 2;fclose(f);uint32_t*train=malloc(h.rows*4),*test=malloc(h.rows*4),nt=0,nv=0;if(!train||!test)return 2;for(uint32_t i=0;i<h.rows;i++)(r[i].test?test:train)[r[i].test?nv++:nt++]=i;*records=r;*train_out=train;*nt_out=nt;*test_out=test;*nv_out=nv;return 0;}
static int run_multistep(const char*path,int fused,int probe){Record*r;uint32_t*train,*test,nt,nv;if(load_records(path,&r,&train,&nt,&test,&nv))return 2;OracleTrainingState oracle;oracle_training_init(&oracle);Model canonical;if(model_init(&canonical))return 2;uint32_t local_rng=0x47554e31u,max_ulp=0;float max_abs=0,max_fused=0,max_unfused=0;unsigned max_epoch=0,max_sample=0,nonzero=0;
    for(unsigned epoch=0;epoch<2500;epoch++){
        for(uint32_t i=nt-1;i;i--){local_rng=local_rng*1664525u+1013904223u;uint32_t j=local_rng%(i+1),x=train[i];train[i]=train[j];train[j]=x;}
        for(unsigned position=0;position<nt;position++){
            Snapshot a={0},b={0};Record*sample=&r[train[position]];
            if(fused&&compare_state(epoch,position,&oracle,&canonical))return 1;
            oracle_training_step(&oracle,sample,&a);
            if(fused)canonical_training_step_fused(&canonical,sample,0,&b);else canonical_training_step(&canonical,sample,&b);
            if(probe){float p=b.probability,y=sample->y,one_minus=1.f-p;float fused_grad=p-y;float bce_grad=(p-y)/(p*one_minus);float unfused_grad=bce_grad*p*one_minus;float delta=fabsf(fused_grad-unfused_grad);uint32_t ulp=ulp_distance(fused_grad,unfused_grad);if(delta!=0)nonzero++;if(delta>max_abs||(delta==max_abs&&ulp>max_ulp)){max_abs=delta;max_ulp=ulp;max_fused=fused_grad;max_unfused=unfused_grad;max_epoch=epoch;max_sample=position;}}
            if(!probe){if(fused){CMP_AT(input);CMP_AT(qkv);CMP_AT(prob);CMP_AT(attn);CMP_AT(merged);CMP_AT(proj);CMP_AT(sum1);CMP_AT(ln1);CMP_AT(ff);CMP_AT(sum2);CMP_AT(out);CMP_AT(pooled);}if(compare_at(epoch,position,"loss",&a.loss,&b.loss,1))return 1;if(compare_at(epoch,position,"classifier_logit",&a.logit,&b.logit,1)){if(fused)state_max_report(epoch,position,&oracle,&canonical);return 1;}if(compare_at(epoch,position,"probability",&a.probability,&b.probability,1))return 1;CMP_AT(dwq);CMP_AT(dwo);CMP_AT(dw1);CMP_AT(dw2);CMP_AT(dhead);if(compare_at(epoch,position,"dbias",&a.dbias,&b.dbias,1))return 1;CMP_AT(next_wq);CMP_AT(next_wo);CMP_AT(next_w1);CMP_AT(next_w2);CMP_AT(next_head);if(compare_at(epoch,position,"next_bias",&a.next_bias,&b.next_bias,1))return 1;}
            if(probe&&epoch==203&&position==16){printf("gradient probe: updates=%u nonzero=%u max_abs=%.9g max_ulp=%u at_epoch=%u at_sample=%u fused=%.9g unfused=%.9g\n",epoch*nt+position+1,nonzero,max_abs,max_ulp,max_epoch,max_sample,max_fused,max_unfused);free(test);free(train);free(r);return 0;}
        }
    }
    printf("gunpoint multi-step differential passed: mode=%s epochs=2500 updates=%u\n",fused?"fused":"unfused",2500*nt);free(test);free(train);free(r);return 0;}

static int run_fused_training(const char*path,int shuffled){Record*r;uint32_t*train,*test,nt,nv;if(load_records(path,&r,&train,&nt,&test,&nv))return 2;Model m;if(model_init(&m))return 2;uint32_t local_rng=0x47554e31u;float initial=evaluate(&m,r,test,nv,shuffled);for(unsigned epoch=0;epoch<2500;epoch++){for(uint32_t i=nt-1;i;i--){local_rng=local_rng*1664525u+1013904223u;uint32_t j=local_rng%(i+1),x=train[i];train[i]=train[j];train[j]=x;}for(uint32_t i=0;i<nt;i++){Snapshot ignored={0};canonical_training_step_fused(&m,&r[train[i]],shuffled,&ignored);}}float final=evaluate(&m,r,test,nv,shuffled);printf("gunpoint fused differential experiment: mode=%s initial=%.3f final=%.3f epochs=2500\n",shuffled?"shuffled":"ordered",initial,final);free(test);free(train);free(r);return 0;}

static int run_stability_trace(const char*path,const char*csv_path){Record*r;uint32_t*train,*test,nt,nv;if(load_records(path,&r,&train,&nt,&test,&nv))return 2;FILE*csv=fopen(csv_path,"w");if(!csv)return 2;fprintf(csv,"update,epoch,sample,state_max_abs,state_max_ulp,logit_abs,first_numerical,material_divergence\n");OracleTrainingState oracle;oracle_training_init(&oracle);Model canonical;if(model_init(&canonical))return 2;uint32_t local_rng=0x47554e31u,update=0;int first_numerical=0,first_material=0;unsigned numerical_update=0,material_update=0;
    for(unsigned epoch=0;epoch<2500;epoch++){for(uint32_t i=nt-1;i;i--){local_rng=local_rng*1664525u+1013904223u;uint32_t j=local_rng%(i+1),x=train[i];train[i]=train[j];train[j]=x;}for(unsigned position=0;position<nt;position++){float state_abs;uint32_t state_ulp;state_max_values(&oracle,&canonical,&state_abs,&state_ulp);int numerical_event=!first_numerical&&state_ulp;if(numerical_event){first_numerical=1;numerical_update=update+1;}Snapshot a={0},b={0};Record*sample=&r[train[position]];oracle_training_step(&oracle,sample,&a);canonical_training_step_fused(&canonical,sample,0,&b);update++;float logit_abs=fabsf(a.logit-b.logit);int material=logit_abs>3e-5f*fmaxf(1.f,fmaxf(fabsf(a.logit),fabsf(b.logit)));if(material&&!first_material){first_material=1;material_update=update;}if(update==1||numerical_event||update%250==0||(material&&material_update==update))fprintf(csv,"%u,%u,%u,%.9g,%u,%.9g,%d,%d\n",update,epoch,position,state_abs,state_ulp,logit_abs,numerical_event,material);}}
    fclose(csv);printf("gunpoint stability trace written: path=%s updates=%u first_numerical_update=%u first_material_update=%u canonical_ordered=%.3f\n",csv_path,update,numerical_update,material_update,evaluate(&canonical,r,test,nv,0));free(test);free(train);free(r);return first_numerical&&first_material?0:1;}

static int compare(const char *name,const float*a,const float*b,size_t n){for(size_t i=0;i<n;i++){float tol=3e-5f*fmaxf(1.f,fmaxf(fabsf(a[i]),fabsf(b[i])));if(fabsf(a[i]-b[i])>tol){printf("FIRST_DIVERGENCE stage=%s index=%zu oracle=%.9g canonical=%.9g delta=%.9g\n",name,i,a[i],b[i],b[i]-a[i]);return 1;}}printf("match stage=%s elements=%zu\n",name,n);return 0;}
#define CMP(field) do{if(compare(#field,a.field,b.field,sizeof a.field/sizeof(float)))return 1;}while(0)
int main(int argc,char**argv){if(argc==4&&!strcmp(argv[2],"--stability-trace"))return run_stability_trace(argv[1],argv[3]);if(argc==3){if(!strcmp(argv[2],"--multi"))return run_multistep(argv[1],0,0);if(!strcmp(argv[2],"--probe"))return run_multistep(argv[1],0,1);if(!strcmp(argv[2],"--multi-fused"))return run_multistep(argv[1],1,0);if(!strcmp(argv[2],"--train-fused-ordered"))return run_fused_training(argv[1],0);if(!strcmp(argv[2],"--train-fused-shuffled"))return run_fused_training(argv[1],1);}if(argc!=2){fprintf(stderr,"usage: %s sequence.bin [--multi|--probe|--multi-fused|--train-fused-ordered|--train-fused-shuffled|--stability-trace output.csv]\n",argv[0]);return 2;}FILE*f=fopen(argv[1],"rb");if(!f)return 2;Header h;if(exact(&h,sizeof h,f))return 2;Record r;if(exact(&r.test,4,f)||exact(&r.id,4,f)||exact(r.x,sizeof r.x,f)||exact(&r.y,4,f))return 2;fclose(f);Snapshot a={0},b={0};oracle_snapshot(&r,&a);canonical_snapshot(&r,&b);
    CMP(input);CMP(qkv);CMP(prob);CMP(attn);CMP(merged);CMP(proj);CMP(sum1);CMP(ln1);CMP(ff);CMP(sum2);CMP(out);CMP(pooled);
    if(compare("classifier_logit",&a.logit,&b.logit,1))return 1;if(compare("probability",&a.probability,&b.probability,1))return 1;if(compare("loss",&a.loss,&b.loss,1))return 1;
    CMP(dwq);CMP(dwo);CMP(dw1);CMP(dw2);CMP(dhead);if(compare("dbias",&a.dbias,&b.dbias,1))return 1;
    CMP(next_wq);CMP(next_wo);CMP(next_w1);CMP(next_w2);CMP(next_head);if(compare("next_bias",&a.next_bias,&b.next_bias,1))return 1;
    puts("gunpoint differential passed: oracle=canonical");return 0;}
