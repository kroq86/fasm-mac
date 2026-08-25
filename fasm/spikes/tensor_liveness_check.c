#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
 Z1_DATA,Z1_GRAD,Z1B_DATA,Z1B_GRAD,H_DATA,H_GRAD,
 Z2_DATA,Z2_GRAD,PRED_DATA,PRED_GRAD,LOSS_DATA,BUFFER_COUNT
};
static const char *names[BUFFER_COUNT]={
 "z1.data","z1.grad","z1b.data","z1b.grad","h.data","h.grad",
 "z2.data","z2.grad","pred.data","pred.grad","loss.data"
};
static const uint32_t bytes[BUFFER_COUNT]={64,64,64,64,64,64,16,16,16,16,4};

/* Combined training schedule: six forward events then six reverse events. */
static const uint64_t event_uses[12]={
 1ull<<Z1_DATA,
 (1ull<<Z1_DATA)|(1ull<<Z1B_DATA),
 (1ull<<Z1B_DATA)|(1ull<<H_DATA),
 (1ull<<H_DATA)|(1ull<<Z2_DATA),
 (1ull<<Z2_DATA)|(1ull<<PRED_DATA),
 (1ull<<PRED_DATA)|(1ull<<LOSS_DATA),
 (1ull<<PRED_DATA)|(1ull<<PRED_GRAD),
 (1ull<<PRED_GRAD)|(1ull<<Z2_GRAD),
 (1ull<<Z2_GRAD)|(1ull<<H_DATA)|(1ull<<H_GRAD),
 (1ull<<H_GRAD)|(1ull<<Z1B_DATA)|(1ull<<Z1B_GRAD),
 (1ull<<Z1B_GRAD)|(1ull<<Z1_GRAD),
 1ull<<Z1_GRAD
};

int main(void){
 uint8_t first[BUFFER_COUNT],last[BUFFER_COUNT],slot[BUFFER_COUNT];
 memset(first,0xff,sizeof first);memset(last,0,sizeof last);memset(slot,0xff,sizeof slot);
 for(uint8_t e=0;e<12;e++)for(uint8_t b=0;b<BUFFER_COUNT;b++)if(event_uses[e]&(1ull<<b)){
   if(first[b]==0xff)first[b]=e;last[b]=e;
 }
 uint8_t order[BUFFER_COUNT];for(uint8_t b=0;b<BUFFER_COUNT;b++)order[b]=b;
 for(uint8_t i=1;i<BUFFER_COUNT;i++){uint8_t v=order[i],j=i;while(j&&first[order[j-1]]>first[v]){order[j]=order[j-1];j--;}order[j]=v;}
 unsigned slot_count=0;
 for(uint8_t oi=0;oi<BUFFER_COUNT;oi++){
   uint8_t b=order[oi];
   for(unsigned s=0;s<slot_count;s++){
     int conflict=0;
     for(uint8_t p=0;p<BUFFER_COUNT;p++)if(slot[p]==s && !(last[p]<first[b] || last[b]<first[p])){conflict=1;break;}
     if(!conflict){slot[b]=s;break;}
   }
   if(slot[b]==0xff)slot[b]=slot_count++;
 }
 unsigned separate_aligned=BUFFER_COUNT*64,theoretical=slot_count*64,raw_peak=0;
 unsigned separate_bytes=0,gradient_bytes=0,data_peak=0;
 for(uint8_t b=0;b<BUFFER_COUNT;b++){
   separate_bytes+=bytes[b];
   if(b==Z1_GRAD||b==Z1B_GRAD||b==H_GRAD||b==Z2_GRAD||b==PRED_GRAD)
     gradient_bytes+=bytes[b];
 }
 for(uint8_t e=0;e<12;e++){
   unsigned live=0,data_live=0;
   for(uint8_t b=0;b<BUFFER_COUNT;b++)if(first[b]<=e&&e<=last[b]){
     live+=bytes[b];
     if(!(b==Z1_GRAD||b==Z1B_GRAD||b==H_GRAD||b==Z2_GRAD||b==PRED_GRAD))
       data_live+=bytes[b];
   }
   if(live>raw_peak)raw_peak=live;
   if(data_live>data_peak)data_peak=data_live;
 }
 /* Current backward kernels accumulate with +=. Reusing a gradient range
    therefore requires a zero-init plan step at each new lifetime. Until that
    step exists, reuse data ranges but keep gradient ranges unique. */
 unsigned executable=data_peak+gradient_bytes;
 if(first[Z1_DATA]!=0||last[Z1_DATA]!=1||first[Z1_GRAD]!=10||last[Z1_GRAD]!=11)return 1;
 if(first[Z1B_DATA]!=1||last[Z1B_DATA]!=9||first[H_DATA]!=2||last[H_DATA]!=8)return 1;
 if(slot_count!=4||theoretical!=256||separate_aligned!=704||raw_peak!=208)return 1;
 if(separate_bytes!=452||data_peak!=160||gradient_bytes!=224||executable!=384)return 1;
 for(uint8_t b=0;b<BUFFER_COUNT;b++)
   printf("%-10s first=%u last=%u scratch_offset=%u\n",names[b],first[b],last[b],slot[b]*64);
 printf("liveness spike passed: separate_bytes=%u executable=%u theoretical_lower_bound=%u raw_peak=%u data_peak=%u grad_unique=%u savings=%.2fx\n",
        separate_bytes,executable,theoretical,raw_peak,data_peak,gradient_bytes,
        (double)separate_bytes/executable);
 return 0;
}
