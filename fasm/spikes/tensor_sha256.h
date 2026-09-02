#ifndef TENSOR_SHA256_H
#define TENSOR_SHA256_H
/* Minimal self-written SHA-256 (FIPS 180-4), no external library. Used to
 * verify a downloaded model artifact's fingerprint as part of the gate
 * itself, not as a manually-written note. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; uint32_t buflen; } Sha256;

static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};
static uint32_t sha256_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_block(Sha256 *s, const uint8_t *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = ((uint32_t)p[i*4]<<24)|((uint32_t)p[i*4+1]<<16)|((uint32_t)p[i*4+2]<<8)|p[i*4+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = sha256_rotr(w[i-15],7) ^ sha256_rotr(w[i-15],18) ^ (w[i-15]>>3);
        uint32_t s1 = sha256_rotr(w[i-2],17) ^ sha256_rotr(w[i-2],19) ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = sha256_rotr(e,6) ^ sha256_rotr(e,11) ^ sha256_rotr(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = sha256_rotr(a,2) ^ sha256_rotr(a,13) ^ sha256_rotr(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d; s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}

static void sha256_init(Sha256 *s) {
    static const uint32_t iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(s->h, iv, sizeof iv);
    s->len = 0; s->buflen = 0;
}
static void sha256_update(Sha256 *s, const uint8_t *data, size_t n) {
    s->len += n;
    while (n > 0) {
        size_t take = 64 - s->buflen; if (take > n) take = n;
        memcpy(s->buf + s->buflen, data, take);
        s->buflen += (uint32_t)take; data += take; n -= take;
        if (s->buflen == 64) { sha256_block(s, s->buf); s->buflen = 0; }
    }
}
static void sha256_final_hex(Sha256 *s, char out_hex[65]) {
    uint64_t bitlen = s->len * 8;
    uint8_t pad = 0x80;
    sha256_update(s, &pad, 1);
    uint8_t zero = 0;
    while (s->buflen != 56) sha256_update(s, &zero, 1);
    uint8_t lenbytes[8];
    for (int i = 0; i < 8; i++) lenbytes[i] = (uint8_t)(bitlen >> (56 - 8*i));
    /* bypass sha256_update's length accounting for the length field itself */
    memcpy(s->buf + 56, lenbytes, 8);
    sha256_block(s, s->buf);
    static const char *hexd = "0123456789abcdef";
    for (int i = 0; i < 8; i++) {
        uint32_t v = s->h[i];
        for (int j = 0; j < 4; j++) {
            uint8_t byte = (uint8_t)(v >> (24 - 8*j));
            out_hex[(i*4+j)*2] = hexd[byte >> 4];
            out_hex[(i*4+j)*2+1] = hexd[byte & 0xf];
        }
    }
    out_hex[64] = '\0';
}

/* hashes a whole file by path; returns 0 and fills out_hex on success. */
static int sha256_file(const char *path, char out_hex[65]) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    Sha256 s; sha256_init(&s);
    uint8_t buf[1 << 16];
    size_t got;
    while ((got = fread(buf, 1, sizeof buf, f)) > 0) sha256_update(&s, buf, got);
    fclose(f);
    sha256_final_hex(&s, out_hex);
    return 0;
}

#endif
