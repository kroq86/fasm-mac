#ifndef TENSOR_SHA256_H
#define TENSOR_SHA256_H
/* SHA-256 (FIPS 180-4) via Apple's CommonCrypto, hardware-accelerated on
 * both arm64 and x86_64. Previously a minimal self-written, unaccelerated
 * scalar implementation (see git history); that version was measured to
 * dominate roughly 85% of native tensor-kv-handoff wall-clock time when
 * fingerprinting large (100s of MB) safetensors bundles on every cold
 * invocation (scratchpad/kv_transfer_audit_20260913/sha256_isolation_
 * bench.c: ~4.4s of a ~5.1s arm64 run). This version hashes the same
 * 876 MB in ~0.55s on arm64. Same public function signatures and
 * semantics as the version it replaces; output is byte-identical (SHA-256
 * is a deterministic standard), verified against tensor_sha256_
 * differential_check.c's independent Python-hashlib oracle vectors and
 * against known-correct fingerprints of the kv-handoff bundle files. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
/* Callers of this header (e.g. tensor_kv_handoff.c's build) pass single-
 * letter -D macros for tensor dims (D=head_dim, H=n_heads, M=d_model, ...).
 * CommonDigest.h's unrelated MD4/MD5 struct definitions use `D` as a plain
 * field name (CC_LONG A,B,C,D;), so a textual macro substitution there is a
 * hard compile error. Shadow the macro only for this include. */
#pragma push_macro("D")
#undef D
#include <CommonCrypto/CommonDigest.h>
#pragma pop_macro("D")

typedef CC_SHA256_CTX Sha256;

static void sha256_init(Sha256 *s) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    CC_SHA256_Init(s);
#pragma clang diagnostic pop
}
static void sha256_update(Sha256 *s, const uint8_t *data, size_t n) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    CC_SHA256_Update(s, data, (CC_LONG)n);
#pragma clang diagnostic pop
}
static void sha256_final_hex(Sha256 *s, char out_hex[65]) {
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    CC_SHA256_Final(digest, s);
#pragma clang diagnostic pop
    static const char *hexd = "0123456789abcdef";
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
        out_hex[i * 2] = hexd[digest[i] >> 4];
        out_hex[i * 2 + 1] = hexd[digest[i] & 0xf];
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
