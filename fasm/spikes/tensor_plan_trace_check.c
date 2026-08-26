#include "tensor_plan_trace_spike.h"
#include <stdlib.h>
int main(int argc,char**argv){if(argc!=2)return 2;PlanTrace trace={0};PlanAlternative memory[]={{"save",5440,0,21,1},{"rematerialize",4928,64,22,1}};
    if(trace_choose_memory(&trace,"attention_scores",memory,2,5000,8192))return 1;PlanDecisionTrace*d=&trace.decisions[0];if(strcmp(d->alternatives[d->chosen].name,"rematerialize")||d->counterfactual_choice!=0||trace_verify_actions(d,22)||!trace_verify_actions(d,21))return 2;
    PlanAlternative layout[]={{"standard",5760,174,21,1},{"layout-aware",5712,172,20,1}};MeasurementProvenance p={"measured","apple-m1","clang-17","release-O2","9b9cb08",101,172.0,168.0,179.0,1};
    if(trace_add_measured(&trace,"head_merge_layout",layout,2,1,p,"lower median; difference within noise"))return 3;if(trace_verify_actions(&trace.decisions[1],20))return 4;
    PlanAlternative neon[]={{"auto-vectorized-c",0,53,1,1},{"manual-neon",0,126,1,1}};p=(MeasurementProvenance){"measured","apple-m1","clang-17","release-O2","da88be8",101,53,51,58,0};if(trace_add_measured(&trace,"relu_kernel",neon,2,0,p,"manual NEON is slower on this shape"))return 5;
    FILE*f=fopen(argv[1],"wb");if(!f||trace_export_tsv(f,&trace)||fclose(f))return 6;f=fopen(argv[1],"rb");if(!f)return 7;char text[8192];size_t n=fread(text,1,sizeof text-1,f);text[n]=0;fclose(f);if(!strstr(text,"attention_scores\trematerialize\t1")||!strstr(text,"manual NEON is slower")||!strstr(text,"apple-m1\tclang-17"))return 8;
    f=tmpfile();if(!f||trace_explain(f,&trace)||fseek(f,0,SEEK_SET))return 9;n=fread(text,1,sizeof text-1,f);text[n]=0;fclose(f);if(!strstr(text,"chosen: rematerialize")||!strstr(text,"counterfactual budget=8192 -> save")||!strstr(text,"confidence=uncertain"))return 10;
    printf("plan decision trace passed: decisions=%u alternatives=6 counterfactual=save@8192 provenance=measured negative_result=manual-neon consistency=actions-match export=tsv\n",trace.count);return 0;}
