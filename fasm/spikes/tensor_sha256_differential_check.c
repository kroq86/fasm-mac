/* Differential check for the self-written SHA-256 (tensor_sha256.h),
 * used to verify downloaded model artifact fingerprints as part of the
 * GPT-2 block0 gate itself. Oracle values are Python's hashlib (an
 * independent, standard-library implementation), covering the empty
 * message, a short message, an exact single-block-boundary case (56
 * bytes: 55 data + 0x80 pad byte land exactly at the 56-byte length-field
 * boundary), and a multi-block message (129 bytes) that exercises the
 * padding/length-field logic across a block split.
 */
#include "tensor_sha256.h"
#include <stdio.h>
#include <string.h>

static int check(const char *label, const uint8_t *msg, size_t len, const char *expect) {
    Sha256 s; char hex[65];
    sha256_init(&s);
    sha256_update(&s, msg, len);
    sha256_final_hex(&s, hex);
    if (strcmp(hex, expect)) { fprintf(stderr, "%s: got %s expected %s\n", label, hex, expect); return 1; }
    printf("%s: matches independent oracle (%s)\n", label, hex);
    return 0;
}

int main(void) {
    if (check("empty", (const uint8_t *)"", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")) return 1;
    if (check("abc", (const uint8_t *)"abc", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")) return 1;

    const char *m56 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    if (check("56byte_block_boundary", (const uint8_t *)m56, strlen(m56), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")) return 1;

    uint8_t big[129]; for (int i = 0; i < 129; i++) big[i] = 'a';
    if (check("129byte_multiblock", big, 129, "c12cb024a2e5551cca0e08fce8f1c5e314555cc3fef6329ee994a3db752166ae")) return 1;

    /* incremental updates (many small chunks) must equal one bulk update --
     * exercises the internal buffer/carry logic independent of chunking */
    Sha256 s1, s2; char hex1[65], hex2[65];
    uint8_t msg[200]; for (int i = 0; i < 200; i++) msg[i] = (uint8_t)(i * 37 + 11);
    sha256_init(&s1); sha256_update(&s1, msg, 200); sha256_final_hex(&s1, hex1);
    sha256_init(&s2);
    for (int off = 0; off < 200; ) {
        int chunk = 1 + (off % 7);
        if (off + chunk > 200) chunk = 200 - off;
        sha256_update(&s2, msg + off, (size_t)chunk);
        off += chunk;
    }
    sha256_final_hex(&s2, hex2);
    if (strcmp(hex1, hex2)) { fprintf(stderr, "chunking mismatch: bulk=%s incremental=%s\n", hex1, hex2); return 1; }
    printf("chunking_invariance: bulk update and many small incremental updates agree exactly (%s)\n", hex1);

    puts("sha256 differential check passed: matches Python hashlib on empty/short/block-boundary/multi-block messages, chunking-invariant");
    return 0;
}
