#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct{const char*name;uint32_t bytes;uint8_t first,last,alias;uint32_t offset;}Buffer;
#define NO_ALIAS 255
static uint32_t align64(uint32_t n){return(n+63)&~63u;}
static int overlap(const Buffer*a,const Buffer*b){return!(a->last<b->first||b->last<a->first);}
int main(void){/* Events 0..13 forward; 14..22 zero; 23..34 reverse/remat. */
 Buffer b[]={
  {"packed_qkv",768,0,34,NO_ALIAS,0},{"q_view",0,0,34,0,0},{"k_view",0,0,34,0,0},{"v_view",0,0,34,0,0},
  {"scores_fwd",512,4,4,NO_ALIAS,0},{"prob",512,4,33,NO_ALIAS,0},{"head",256,4,31,NO_ALIAS,0},{"merged",256,5,30,NO_ALIAS,0},
  {"projected",256,6,30,NO_ALIAS,0},{"sum1",256,7,29,NO_ALIAS,0},{"ln1",256,8,29,NO_ALIAS,0},{"ln1_stats",64,8,28,NO_ALIAS,0},
  {"ff1",512,9,27,NO_ALIAS,0},{"activation",512,10,25,NO_ALIAS,0},{"ff2",256,11,25,NO_ALIAS,0},{"sum2",256,12,23,NO_ALIAS,0},
  {"output",256,13,23,NO_ALIAS,0},{"ln2_stats",64,13,23,NO_ALIAS,0},{"d_sum2",256,23,24,NO_ALIAS,0},{"d_ln1",256,24,29,NO_ALIAS,0},
  {"d_ff2",256,24,25,NO_ALIAS,0},{"d_act",512,25,26,NO_ALIAS,0},{"d_ff1",512,26,27,NO_ALIAS,0},{"d_sum1",256,28,29,NO_ALIAS,0},
  {"d_projected",256,29,30,NO_ALIAS,0},{"d_merged",256,30,31,NO_ALIAS,0},{"d_head",256,31,33,NO_ALIAS,0},{"scores_remat",512,32,33,NO_ALIAS,0},
  {"d_prob",512,33,33,NO_ALIAS,0},{"d_score",512,33,33,NO_ALIAS,0},{"d_qkv",768,33,34,NO_ALIAS,0}};
 const unsigned count=sizeof b/sizeof b[0];uint32_t arena_end=0,separate=0;for(unsigned i=0;i<count;i++){if(b[i].alias!=NO_ALIAS){b[i].offset=b[b[i].alias].offset;continue;}uint32_t candidate=0;for(;;){int conflict=0;for(unsigned j=0;j<i;j++)if(b[j].alias==NO_ALIAS&&overlap(&b[i],&b[j])){uint32_t je=b[j].offset+align64(b[j].bytes);if(candidate<je&&candidate+align64(b[i].bytes)>b[j].offset){candidate=je;conflict=1;break;}}if(!conflict)break;}b[i].offset=candidate;uint32_t end=candidate+align64(b[i].bytes);if(end>arena_end)arena_end=end;separate+=align64(b[i].bytes);}
 uint32_t raw_peak=0;for(unsigned e=0;e<35;e++){uint32_t live=0;for(unsigned i=0;i<count;i++)if(b[i].alias==NO_ALIAS&&b[i].first<=e&&e<=b[i].last)live+=b[i].bytes;if(live>raw_peak)raw_peak=live;}
 if(b[1].offset!=b[0].offset||b[2].offset!=b[0].offset||b[3].offset!=b[0].offset)return 1;if(arena_end>=5760||raw_peak>arena_end)return 1;
 printf("tensor transformer liveness passed: events=35 buffers=%u aliases=3 separate_aligned=%u raw_peak=%u arena_bytes=%u upper_bound=5760 savings=%.2fx views_allocate=no remat_scores=split-interval\n",count,separate,raw_peak,arena_end,(double)separate/arena_end);return 0;}
