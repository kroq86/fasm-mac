#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CKPT_OK, CKPT_IO, CKPT_FORMAT, CKPT_SHAPE, CKPT_CHECKSUM };
enum { CKPT_F32 = 1, CKPT_MAX_TENSORS = 8, CKPT_MAX_RANK = 4 };

typedef struct {
    char name[16];
    uint32_t dtype, rank, dims[CKPT_MAX_RANK], count;
    float *data;
} Parameter;

typedef struct {
    char magic[8];
    uint32_t version, tensor_count, payload_bytes, checksum;
} CheckpointHeader;

typedef struct {
    char name[16];
    uint32_t dtype, rank, dims[CKPT_MAX_RANK], count;
} TensorHeader;

static uint32_t checksum_update(uint32_t h, const void *ptr, size_t n) {
    const unsigned char *p = ptr;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static uint32_t payload_checksum(const Parameter *p, uint32_t n) {
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < n; i++) {
        TensorHeader th = {0};
        memcpy(th.name, p[i].name, sizeof th.name);
        th.dtype = p[i].dtype; th.rank = p[i].rank; th.count = p[i].count;
        memcpy(th.dims, p[i].dims, sizeof th.dims);
        h = checksum_update(h, &th, sizeof th);
        h = checksum_update(h, p[i].data, p[i].count * sizeof(float));
    }
    return h;
}

static int checkpoint_save(const char *path, const Parameter *p, uint32_t n) {
    if (!n || n > CKPT_MAX_TENSORS) return CKPT_FORMAT;
    uint32_t bytes = 0;
    for (uint32_t i = 0; i < n; i++) bytes += sizeof(TensorHeader) + p[i].count * sizeof(float);
    CheckpointHeader h = {{'F','A','S','M','T','N','S','R'}, 1, n, bytes, payload_checksum(p, n)};
    FILE *f = fopen(path, "wb");
    if (!f) return CKPT_IO;
    int ok = fwrite(&h, sizeof h, 1, f) == 1;
    for (uint32_t i = 0; ok && i < n; i++) {
        TensorHeader th = {0};
        memcpy(th.name, p[i].name, sizeof th.name);
        th.dtype = p[i].dtype; th.rank = p[i].rank; th.count = p[i].count;
        memcpy(th.dims, p[i].dims, sizeof th.dims);
        ok = fwrite(&th, sizeof th, 1, f) == 1 &&
             fwrite(p[i].data, sizeof(float), p[i].count, f) == p[i].count;
    }
    ok = ok && fclose(f) == 0;
    return ok ? CKPT_OK : CKPT_IO;
}

static int checkpoint_load(const char *path, Parameter *expected, uint32_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return CKPT_IO;
    CheckpointHeader h;
    if (fread(&h, sizeof h, 1, f) != 1 || memcmp(h.magic, "FASMTNSR", 8) ||
        h.version != 1 || h.tensor_count != n) { fclose(f); return CKPT_FORMAT; }
    uint32_t hash = 2166136261u, bytes = 0;
    for (uint32_t i = 0; i < n; i++) {
        TensorHeader th;
        if (fread(&th, sizeof th, 1, f) != 1) { fclose(f); return CKPT_IO; }
        bytes += sizeof th + th.count * sizeof(float);
        if (memcmp(th.name, expected[i].name, sizeof th.name) || th.dtype != CKPT_F32 ||
            th.rank != expected[i].rank || th.count != expected[i].count ||
            memcmp(th.dims, expected[i].dims, sizeof th.dims)) { fclose(f); return CKPT_SHAPE; }
        hash = checksum_update(hash, &th, sizeof th);
        if (fread(expected[i].data, sizeof(float), th.count, f) != th.count) { fclose(f); return CKPT_IO; }
        hash = checksum_update(hash, expected[i].data, th.count * sizeof(float));
    }
    int trailing = fgetc(f), close_error = fclose(f);
    if (close_error || trailing != EOF || bytes != h.payload_bytes) return CKPT_FORMAT;
    return hash == h.checksum ? CKPT_OK : CKPT_CHECKSUM;
}

static int corrupt_last_byte(const char *path) {
    FILE *f = fopen(path, "r+b");
    if (!f || fseek(f, -1, SEEK_END)) return 0;
    int c = fgetc(f);
    if (c == EOF || fseek(f, -1, SEEK_CUR)) { fclose(f); return 0; }
    unsigned char changed = (unsigned char)c ^ 0x80u;
    int ok = fwrite(&changed, 1, 1, f) == 1 && fclose(f) == 0;
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    float wq[12], wo[4], loaded_wq[12] = {0}, loaded_wo[4] = {0};
    for (int i = 0; i < 12; i++) wq[i] = (float)(i - 5) * .125f;
    for (int i = 0; i < 4; i++) wo[i] = (float)(i + 1) * -.25f;
    Parameter source[] = {{"wq", CKPT_F32, 2, {3,4}, 12, wq}, {"wo", CKPT_F32, 2, {2,2}, 4, wo}};
    Parameter target[] = {{"wq", CKPT_F32, 2, {3,4}, 12, loaded_wq}, {"wo", CKPT_F32, 2, {2,2}, 4, loaded_wo}};
    if (checkpoint_save(argv[1], source, 2) != CKPT_OK || checkpoint_load(argv[1], target, 2) != CKPT_OK ||
        memcmp(wq, loaded_wq, sizeof wq) || memcmp(wo, loaded_wo, sizeof wo)) return 3;
    Parameter wrong[] = {{"wq", CKPT_F32, 2, {4,3}, 12, loaded_wq}, {"wo", CKPT_F32, 2, {2,2}, 4, loaded_wo}};
    if (checkpoint_load(argv[1], wrong, 2) != CKPT_SHAPE) return 4;
    if (!corrupt_last_byte(argv[1]) || checkpoint_load(argv[1], target, 2) != CKPT_CHECKSUM) return 5;
    printf("tensor checkpoint spike passed: version=1 dtype=f32 tensors=2 roundtrip=exact shape_mismatch=rejected corruption=rejected\n");
    return 0;
}
