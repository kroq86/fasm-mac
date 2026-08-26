#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { MAX_TENSORS = 8 };
typedef struct { char name[16]; const unsigned char *data; uint32_t bytes, checksum; } Record;
typedef struct { uint32_t valid_mask, corrupt_mask, first_corrupt; char first_name[16]; } Report;

static uint32_t checksum(const unsigned char *p, uint32_t n) {
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static Report inspect(const Record *records, uint32_t count) {
    Report report = {.first_corrupt=UINT32_MAX};
    for (uint32_t i = 0; i < count; i++) {
        if (checksum(records[i].data, records[i].bytes) == records[i].checksum) {
            report.valid_mask |= 1u << i;
        } else {
            report.corrupt_mask |= 1u << i;
            if (report.first_corrupt == UINT32_MAX) {
                report.first_corrupt = i;
                memcpy(report.first_name, records[i].name, sizeof report.first_name);
            }
        }
    }
    return report;
}

/* Normal model loading is transactional: diagnostics may identify healthy
   records, but no destination is modified unless every requested record is
   valid. */
static int load_all_or_nothing(const Record *records, uint32_t count,
                              unsigned char *const *destinations) {
    Report report = inspect(records, count);
    if (report.corrupt_mask) return -1;
    for (uint32_t i = 0; i < count; i++) memcpy(destinations[i], records[i].data, records[i].bytes);
    return 0;
}

/* Explicit repair tooling may recover individually verified tensors. */
static uint32_t recover_valid(const Record *records, uint32_t count,
                              unsigned char *const *destinations) {
    Report report = inspect(records, count);
    for (uint32_t i = 0; i < count; i++)
        if (report.valid_mask & (1u << i)) memcpy(destinations[i], records[i].data, records[i].bytes);
    return report.valid_mask;
}

int main(void) {
    unsigned char wq[] = {1,2,3,4,5,6}, wo[] = {9,8,7,6}, velocity[] = {3,1,4};
    Record records[] = {{"wq",wq,sizeof wq,checksum(wq,sizeof wq)},
                        {"wo",wo,sizeof wo,checksum(wo,sizeof wo)},
                        {"optimizer.v",velocity,sizeof velocity,checksum(velocity,sizeof velocity)}};
    unsigned char out_wq[sizeof wq], out_wo[sizeof wo], out_v[sizeof velocity];
    unsigned char *out[] = {out_wq,out_wo,out_v};
    memset(out_wq,0xaa,sizeof out_wq); memset(out_wo,0xaa,sizeof out_wo); memset(out_v,0xaa,sizeof out_v);
    wo[2] ^= 0x40;
    Report report = inspect(records, 3);
    if (report.valid_mask != 0x5 || report.corrupt_mask != 0x2 || report.first_corrupt != 1 || strcmp(report.first_name,"wo")) return 1;
    if (load_all_or_nothing(records,3,out) != -1) return 2;
    for (uint32_t i=0;i<sizeof out_wq;i++) if(out_wq[i]!=0xaa) return 3;
    for (uint32_t i=0;i<sizeof out_wo;i++) if(out_wo[i]!=0xaa) return 4;
    for (uint32_t i=0;i<sizeof out_v;i++) if(out_v[i]!=0xaa) return 5;
    if (recover_valid(records,3,out) != 0x5 || memcmp(out_wq,wq,sizeof wq) || memcmp(out_v,velocity,sizeof velocity)) return 6;
    for (uint32_t i=0;i<sizeof out_wo;i++) if(out_wo[i]!=0xaa) return 7;
    wo[2] ^= 0x40;
    if (load_all_or_nothing(records,3,out) || memcmp(out_wo,wo,sizeof wo)) return 8;
    puts("tensor checkpoint integrity spike passed: per_tensor_checksum=yes corrupt=wo intact=wq,optimizer.v normal_load=all-or-nothing explicit_recovery=valid-only");
    return 0;
}
