/* tensorctl build: lower the imported ONNX v0 graph to a standalone native
 * executable. The artifact accepts INPUT.f32 and OUTPUT.f32 paths; weights are
 * embedded from ONNX initializers without sidecar files or model-specific code. */
#include "tensor_onnx_import.h"
#include <limits.h>
#include <sys/wait.h>
#include <unistd.h>

static uint64_t build_elems(const int64_t*d,uint32_t n){uint64_t x=1;for(uint32_t i=0;i<n;i++){if(d[i]<=0)return 0;x*=(uint64_t)d[i];}return x;}
static int initializer_index(const OnnxGraph*g,const char*name){for(uint32_t i=0;i<g->n_initializers;i++)if(!strcmp(g->initializers[i].name,name))return(int)i;return-1;}
static int input_index(const OnnxGraph*g,const char*name){for(uint32_t i=0;i<g->n_inputs;i++)if(!strcmp(g->inputs[i].name,name))return(int)i;return-1;}
/* Older ONNX IR (ir_version=3-era exports, e.g. this project's real
 * mnist-8.onnx CNN fixture) requires every initializer to ALSO be listed
 * as a graph input -- g->n_inputs counts those duplicates too, so
 * g->n_inputs!=1 is not "more than one real input", it's "any
 * initializer-shadowed entries at all". Find the one input that is NOT
 * also an initializer -- that is the actual runtime input this build
 * targets. Same convention already used by
 * onnx_validate_storage_and_types() in tensor_onnx_import.h. */
static int real_input_index(const OnnxGraph*g){int found=-1;for(uint32_t i=0;i<g->n_inputs;i++){if(initializer_index(g,g->inputs[i].name)>=0)continue;if(found>=0)return-2;found=(int)i;}return found;}
static int producer_index(const OnnxGraph*g,const char*name){for(uint32_t i=0;i<g->n_nodes;i++)for(uint32_t j=0;j<g->nodes[i].n_outputs;j++)if(!strcmp(g->nodes[i].outputs[j],name))return(int)i;return-1;}
/* INT64 initializers (Reshape's target-shape operand) are pure shape
 * metadata: onnx_infer_shapes() already resolved them into n->out_dims at
 * import time, and no codegen path here ever references them by w%d --
 * only FLOAT initializers actually get embedded as weight arrays and
 * need this check. */
static int build_preflight(const OnnxGraph*g){int bad=0;for(uint32_t i=0;i<g->n_nodes;i++)if(!(onnx_op_capability(g->nodes[i].op_type)&ONNX_CAP_BUILD)){fprintf(stderr,"[unsupported] node '%s' (%s): importable for inspection, but native lowering is not implemented\n",g->nodes[i].name,g->nodes[i].op_type);bad=1;}for(uint32_t i=0;i<g->n_initializers;i++){if(g->initializers[i].data_type==7)continue;if(g->initializers[i].data_type!=1||!g->initializers[i].data){fprintf(stderr,"[unsupported] initializer '%s': native build requires inline float32 data\n",g->initializers[i].name);bad=1;}}return bad?-1:0;}
static int deterministic_uuid(const char*path,const uint8_t*model,size_t model_len){uint64_t h1=1469598103934665603ull,h2=1099511628211ull;for(size_t i=0;i<model_len;i++){h1=(h1^model[i])*1099511628211ull;h2=(h2+model[i])*0x9e3779b185ebca87ull;}FILE*f=fopen(path,"r+b");if(!f)return-1;uint32_t header[8];if(fread(header,4,8,f)!=8||header[0]!=0xfeedfacf){fclose(f);return-1;}long at=32;for(uint32_t i=0;i<header[4];i++){uint32_t command[2];if(fseek(f,at,SEEK_SET)||fread(command,4,2,f)!=2||command[1]<8){fclose(f);return-1;}if(command[0]==0x1b){uint8_t uuid[16];memcpy(uuid,&h1,8);memcpy(uuid+8,&h2,8);uuid[6]=(uuid[6]&15)|64;uuid[8]=(uuid[8]&63)|128;if(fseek(f,at+8,SEEK_SET)||fwrite(uuid,1,16,f)!=16||fclose(f))return-1;return 0;}at+=command[1];}fclose(f);return-1;}
/* True when the real graph input is image-shaped (NCHW, N=1) and this
 * build lowers Conv/MaxPool internally to the project's canonical HWC
 * layout, so the raw file bytes (stored NCHW, matching the ONNX tensor
 * declaration) need one transpose before anything reads them as HWC. */
static int input_is_nchw_image(const OnnxGraph*g,int in_idx){return g->inputs[in_idx].ndim==4&&g->inputs[in_idx].dims[0]==1;}
static void ref(const OnnxGraph*g,int in_idx,FILE*f,const char*name){int i=initializer_index(g,name);if(i>=0){fprintf(f,"w%d",i);return;}if(input_index(g,name)>=0){fputs(input_is_nchw_image(g,in_idx)?"input_hwc":"input",f);return;}int p=producer_index(g,name);fprintf(f,"v%d",p);}

/* Structural (not name-based) detection: a Reshape whose source is a 4-D
 * initializer [C,H,W,K] feeding a MatMul whose OTHER operand traces back
 * (through at most one more Reshape) to a Conv/MaxPool 4-D output with
 * the SAME [C,H,W] is a flatten-then-FC weight stored in NCHW-flatten
 * row order by the exporter -- exactly the "second, less obvious
 * problem" inspect's layout section warns about: this build's
 * activations are HWC-flattened, so these weight rows need permuting to
 * match, or the FC layer silently multiplies the wrong feature against
 * the wrong weight row. Real, previously-diagnosed case, not
 * hypothetical -- this project's own mnist-8.onnx fixture hits it. */
static int weight_needs_nchw_to_hwc_reorder(const OnnxGraph*g,const OnnxNode*rn,int64_t*oc,int64_t*oh,int64_t*ow){
    const OnnxTensor*t=onnx_find_initializer(g,rn->inputs[0]);
    if(!t||t->ndim!=4)return 0;
    int64_t c=t->dims[0],h=t->dims[1],w=t->dims[2];
    for(uint32_t i=0;i<g->n_nodes;i++){
        const OnnxNode*mm=&g->nodes[i];
        if(strcmp(mm->op_type,"MatMul")&&strcmp(mm->op_type,"Gemm"))continue;
        int uses=0;for(uint32_t j=0;j<mm->n_inputs;j++)if(!strcmp(mm->inputs[j],rn->outputs[0]))uses=1;
        if(!uses)continue;
        for(uint32_t j=0;j<mm->n_inputs;j++){
            if(!strcmp(mm->inputs[j],rn->outputs[0]))continue;
            const char*nm=mm->inputs[j];
            for(int hop=0;hop<2;hop++){
                int p=producer_index(g,nm);if(p<0)break;
                const OnnxNode*pn=&g->nodes[p];
                if((!strcmp(pn->op_type,"Conv")||!strcmp(pn->op_type,"MaxPool"))&&pn->out_ndim==4&&pn->out_dims[1]==c&&pn->out_dims[2]==h&&pn->out_dims[3]==w){*oc=c;*oh=h;*ow=w;return 1;}
                if(!strcmp(pn->op_type,"Reshape")){nm=pn->inputs[0];continue;}
                break;
            }
        }
    }
    return 0;
}

static int emit_source(FILE*f,const OnnxGraph*g){if(build_preflight(g))return-1;int in_idx=real_input_index(g);if(in_idx<0||g->n_outputs!=1){fprintf(stderr,"[unsupported] build requires exactly one non-initializer graph input and one graph output\n");return-1;}uint64_t input_n=build_elems(g->inputs[in_idx].dims,g->inputs[in_idx].ndim);if(!input_n){fprintf(stderr,"[unsupported] native build requires fully static positive input dimensions\n");return-1;}
    fputs("#include <math.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\nstatic long bi(long x,const long*od,int on,const long*id,int in){long oi[8]={0},q=x,r=0,m=1;for(int i=on-1;i>=0;i--){oi[i]=q%od[i];q/=od[i];}for(int i=in-1;i>=0;i--){long d=id[i],v=oi[on-in+i];r+=(d==1?0:v)*m;m*=d;}return r;}\n",f);
    for(uint32_t i=0;i<g->n_initializers;i++){const OnnxTensor*t=&g->initializers[i];if(t->data_type==7)continue;/* shape-metadata only, never referenced as w%d */fprintf(f,"static const float w%u[%llu]={",i,(unsigned long long)t->count);for(uint64_t j=0;j<t->count;j++){float x;memcpy(&x,(const uint8_t*)t->data+j*4,4);fprintf(f,"%s%a",j?",":"",(double)x);}fputs("};\n",f);}
    for(uint32_t i=0;i<g->n_nodes;i++)fprintf(f,"static float v%u[%llu];\n",i,(unsigned long long)build_elems(g->nodes[i].out_dims,g->nodes[i].out_ndim));
    fprintf(f,"int main(int argc,char**argv){if(argc!=3){fprintf(stderr,\"usage: %%s INPUT.f32 OUTPUT.f32\\n\",argv[0]);return 2;}float input[%llu];FILE*in=fopen(argv[1],\"rb\");if(!in||fread(input,4,%llu,in)!=%llu||fgetc(in)!=EOF){fprintf(stderr,\"native model: input must contain exactly %llu float32 values\\n\");return 3;}fclose(in);\n",(unsigned long long)input_n,(unsigned long long)input_n,(unsigned long long)input_n,(unsigned long long)input_n);
    if(input_is_nchw_image(g,in_idx)){int64_t C=g->inputs[in_idx].dims[1],H=g->inputs[in_idx].dims[2],W=g->inputs[in_idx].dims[3];fprintf(f,"static float input_hwc[%llu];for(long c=0;c<%lld;c++)for(long h=0;h<%lld;h++)for(long w=0;w<%lld;w++)input_hwc[(h*%lld+w)*%lld+c]=input[(c*%lld+h)*%lld+w];\n",(unsigned long long)input_n,(long long)C,(long long)H,(long long)W,(long long)W,(long long)C,(long long)H,(long long)W);}
    for(uint32_t i=0;i<g->n_nodes;i++){const OnnxNode*n=&g->nodes[i];int64_t da[ONNX_MAX_DIMS],db[ONNX_MAX_DIMS];uint32_t an=0,bn=0;onnx_find_value_shape(g,n->inputs[0],da,&an);if(n->n_inputs>1)onnx_find_value_shape(g,n->inputs[1],db,&bn);
        if(!strcmp(n->op_type,"MatMul")||!strcmp(n->op_type,"Gemm")){int gemm=!strcmp(n->op_type,"Gemm"),ta=gemm&&n->transA,tb=gemm&&n->transB;int64_t rows=ta?da[1]:da[0],inner=ta?da[0]:da[1],cols=tb?db[0]:db[1];fprintf(f,"for(long r=0;r<%lld;r++)for(long c=0;c<%lld;c++){float s=0;for(long k=0;k<%lld;k++)s+=",(long long)rows,(long long)cols,(long long)inner);ref(g,in_idx,f,n->inputs[0]);fputs("[",f);if(ta)fprintf(f,"k*%lld+r",(long long)rows);else fprintf(f,"r*%lld+k",(long long)inner);fputs("] * ",f);ref(g,in_idx,f,n->inputs[1]);fputs("[",f);if(tb)fprintf(f,"c*%lld+k",(long long)inner);else fprintf(f,"k*%lld+c",(long long)cols);fprintf(f,"];v%u[r*%lld+c]=%a*s",i,(long long)cols,(double)(gemm?n->alpha:1.f));if(gemm&&n->n_inputs==3){fprintf(f,"+%a*",(double)n->beta);ref(g,in_idx,f,n->inputs[2]);fputs("[c]",f);}fputs(";}\n",f);
        }else if(!strcmp(n->op_type,"Relu")){uint64_t count=build_elems(n->out_dims,n->out_ndim);fprintf(f,"for(long i=0;i<%llu;i++){float x=",(unsigned long long)count);ref(g,in_idx,f,n->inputs[0]);fprintf(f,"[i];v%u[i]=x>0?x:0;}\n",i);
        }else if(!strcmp(n->op_type,"Add")&&an==4){/* Conv-bias-add on an HWC-STORED 4-D activation: n->out_dims/da describe the
             * ONNX-declared NCHW shape, but the physical buffer here is this build's
             * HWC lowering (same as Conv/MaxPool's own output layout) -- the generic
             * bi() NCHW-broadcast helper below would silently reinterpret an HWC
             * buffer as NCHW-flat and corrupt every value past channel 0. Per-channel
             * bias only, matching this project's CONV convention (bias via a
             * separate Add, indexed by channel) -- this is what a foreign CNN's
             * Conv-bias Add always is once Conv itself has no bias input. */
            int64_t H=da[2],W=da[3],C=da[1];
            fprintf(f,"for(long h=0;h<%lld;h++)for(long w=0;w<%lld;w++)for(long c=0;c<%lld;c++)v%u[(h*%lld+w)*%lld+c]=",(long long)H,(long long)W,(long long)C,i,(long long)W,(long long)C);
            ref(g,in_idx,f,n->inputs[0]);fprintf(f,"[(h*%lld+w)*%lld+c]+",(long long)W,(long long)C);ref(g,in_idx,f,n->inputs[1]);fputs("[c];\n",f);
        }else if(!strcmp(n->op_type,"Add")){uint64_t count=build_elems(n->out_dims,n->out_ndim);fprintf(f,"{static const long od%u[]={",i);for(uint32_t k=0;k<n->out_ndim;k++)fprintf(f,"%s%lld",k?",":"",(long long)n->out_dims[k]);fprintf(f,"},ad%u[]={",i);for(uint32_t k=0;k<an;k++)fprintf(f,"%s%lld",k?",":"",(long long)da[k]);fprintf(f,"},bd%u[]={",i);for(uint32_t k=0;k<bn;k++)fprintf(f,"%s%lld",k?",":"",(long long)db[k]);fprintf(f,"};for(long i=0;i<%llu;i++)v%u[i]=",(unsigned long long)count,i);ref(g,in_idx,f,n->inputs[0]);fprintf(f,"[bi(i,od%u,%u,ad%u,%u)]+",i,n->out_ndim,i,an);ref(g,in_idx,f,n->inputs[1]);fprintf(f,"[bi(i,od%u,%u,bd%u,%u)];}\n",i,n->out_ndim,i,bn);
        }else if(!strcmp(n->op_type,"Conv")){int64_t N0=da[0],CIN=da[1],IH=da[2],IW=da[3],COUT=db[0],KH=db[2],KW=db[3];int64_t OH=n->out_dims[2],OW=n->out_dims[3],SH=n->has_strides?n->strides[0]:1,SW=n->has_strides?n->strides[1]:1,DH=n->dilations[0],DW=n->dilations[1],PT=n->pads[0],PL=n->pads[1];(void)N0;
            fprintf(f,"for(long co=0;co<%lld;co++)for(long oh=0;oh<%lld;oh++)for(long ow=0;ow<%lld;ow++){float s=0;for(long ci=0;ci<%lld;ci++)for(long kh=0;kh<%lld;kh++)for(long kw=0;kw<%lld;kw++){long ih=oh*%lld-%lld+kh*%lld,iw=ow*%lld-%lld+kw*%lld;if(ih>=0&&ih<%lld&&iw>=0&&iw<%lld)s+=",(long long)COUT,(long long)OH,(long long)OW,(long long)CIN,(long long)KH,(long long)KW,(long long)SH,(long long)PT,(long long)DH,(long long)SW,(long long)PL,(long long)DW,(long long)IH,(long long)IW);
            ref(g,in_idx,f,n->inputs[0]);fprintf(f,"[(ih*%lld+iw)*%lld+ci] * ",(long long)IW,(long long)CIN);ref(g,in_idx,f,n->inputs[1]);fprintf(f,"[co*%lld+(ci*%lld+kh)*%lld+kw];}v%u[(oh*%lld+ow)*%lld+co]=s;}\n",(long long)(CIN*KH*KW),(long long)KH,(long long)KW,i,(long long)OW,(long long)COUT);
        }else if(!strcmp(n->op_type,"MaxPool")){int64_t CIN=da[1],IH=da[2],IW=da[3],KH=n->kernel_shape[0],KW=n->kernel_shape[1];int64_t OH=n->out_dims[2],OW=n->out_dims[3],SH=n->has_strides?n->strides[0]:KH,SW=n->has_strides?n->strides[1]:KW,PT=n->pads[0],PL=n->pads[1];
            fprintf(f,"for(long c=0;c<%lld;c++)for(long oh=0;oh<%lld;oh++)for(long ow=0;ow<%lld;ow++){float m=-INFINITY;for(long kh=0;kh<%lld;kh++)for(long kw=0;kw<%lld;kw++){long ih=oh*%lld-%lld+kh,iw=ow*%lld-%lld+kw;if(ih>=0&&ih<%lld&&iw>=0&&iw<%lld){float v=",(long long)CIN,(long long)OH,(long long)OW,(long long)KH,(long long)KW,(long long)SH,(long long)PT,(long long)SW,(long long)PL,(long long)IH,(long long)IW);
            ref(g,in_idx,f,n->inputs[0]);fprintf(f,"[(ih*%lld+iw)*%lld+c];if(v>m)m=v;}}v%u[(oh*%lld+ow)*%lld+c]=m;}\n",(long long)IW,(long long)CIN,i,(long long)OW,(long long)CIN);
        }else if(!strcmp(n->op_type,"Reshape")){uint64_t count=build_elems(n->out_dims,n->out_ndim);int64_t rc,rh,rw;
            if(weight_needs_nchw_to_hwc_reorder(g,n,&rc,&rh,&rw)){int64_t K=(int64_t)count/(rc*rh*rw);
                fprintf(f,"for(long h=0;h<%lld;h++)for(long w=0;w<%lld;w++)for(long c=0;c<%lld;c++)for(long k=0;k<%lld;k++)v%u[((h*%lld+w)*%lld+c)*%lld+k]=",(long long)rh,(long long)rw,(long long)rc,(long long)K,i,(long long)rw,(long long)rc,(long long)K);
                ref(g,in_idx,f,n->inputs[0]);fprintf(f,"[(c*%lld+h)*%lld*%lld+w*%lld+k];\n",(long long)rh,(long long)rw,(long long)K,(long long)K);
            }else{fprintf(f,"for(long i=0;i<%llu;i++)v%u[i]=",(unsigned long long)count,i);ref(g,in_idx,f,n->inputs[0]);fputs("[i];\n",f);}
        }else return-1;
    }
    int out=producer_index(g,g->outputs[0].name);if(out<0)return-1;uint64_t output_n=build_elems(g->outputs[0].dims,g->outputs[0].ndim);fprintf(f,"FILE*out=fopen(argv[2],\"wb\");if(!out||fwrite(v%d,4,%llu,out)!=%llu||fclose(out)){fprintf(stderr,\"native model: output write failed\\n\");return 4;}return 0;}\n",out,(unsigned long long)output_n,(unsigned long long)output_n);return 0;
}

int run_build(int argc,char**argv){if(argc!=4||strcmp(argv[2],"-o")){fprintf(stderr,"usage: tensorctl build MODEL.onnx -o MODEL-NATIVE\n");return 2;}OnnxGraph g;if(onnx_load_file(argv[1],&g))return 3;onnx_validate_storage_and_types(&g);if(!onnx_validate_topology(&g))return 3;onnx_infer_shapes(&g);if(g.n_errors||g.n_unsupported){for(uint32_t i=0;i<g.n_errors;i++)fprintf(stderr,"[error] %s\n",g.errors[i].message);for(uint32_t i=0;i<g.n_unsupported;i++)fprintf(stderr,"[unsupported] %s\n",g.unsupported[i].message);return 4;}char tmp[]="/tmp/tensorctl-native.XXXXXX.c";int fd=mkstemps(tmp,2);if(fd<0)return 3;FILE*f=fdopen(fd,"w");if(!f||emit_source(f,&g)||fclose(f)){unlink(tmp);return 3;}char artifact_tmp[PATH_MAX];if(snprintf(artifact_tmp,sizeof artifact_tmp,"%s.tmp.XXXXXX",argv[3])>=(int)sizeof artifact_tmp){unlink(tmp);return 3;}int artifact_fd=mkstemp(artifact_tmp);if(artifact_fd<0){unlink(tmp);return 3;}close(artifact_fd);unlink(artifact_tmp);pid_t pid=fork();if(pid==0){execlp("clang","clang","-arch","x86_64","-O3",tmp,"-o",artifact_tmp,"-lm",(char*)NULL);_exit(127);}int status=0;if(pid<0||waitpid(pid,&status,0)<0){unlink(tmp);unlink(artifact_tmp);return 3;}unlink(tmp);if(!WIFEXITED(status)||WEXITSTATUS(status)){unlink(artifact_tmp);fprintf(stderr,"tensorctl build: native compiler failed\n");return 3;}if(deterministic_uuid(artifact_tmp,g.file_buf,(size_t)g.file_len)||rename(artifact_tmp,argv[3])){unlink(artifact_tmp);return 3;}printf("[derived] native artifact: %s\n[derived] source: imported ONNX initializers and graph\n[derived] reproducibility: deterministic Mach-O UUID derived from ONNX bytes\n[measured] none (run tensorctl verify)\n",argv[3]);return 0;}
