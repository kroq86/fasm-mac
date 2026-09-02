#ifndef TENSOR_SAFETENSORS_LOADER_H
#define TENSOR_SAFETENSORS_LOADER_H
/* Minimal, self-written safetensors reader: no external library, no
 * PyTorch/Python involved. Format (https://github.com/huggingface/safetensors):
 *   [8 bytes: little-endian u64 header length N]
 *   [N bytes: JSON header -- flat object of tensor_name -> {dtype, shape,
 *    data_offsets:[start,end]}, plus one "__metadata__" entry with no
 *    data_offsets, ignored here]
 *   [raw tensor bytes, byte offsets in data_offsets are relative to the
 *    start of this data section, i.e. absolute_offset = 8 + N + start]
 *
 * This loader validates tensor dtype and shape against the caller's
 * expectation and fails closed (returns nonzero, does not silently guess)
 * on any missing name, dtype mismatch, or shape mismatch -- required by
 * this project's real-block acceptance contract ("fail closed on missing
 * or incompatible data").
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    FILE *f;
    char *header;      /* raw header JSON bytes, NUL-terminated */
    uint64_t data_start; /* absolute file offset where tensor data begins */
    uint64_t file_size;
} SafetensorsFile;

#define ST_MAX_HEADER_BYTES (64u * 1024u * 1024u)

static int st_skip_ws(const char *p) {
    int i = 0;
    while (p[i] == ' ' || p[i] == '\t' || p[i] == '\n' || p[i] == '\r') i++;
    return i;
}

/* copies the JSON string literal starting at *p (which must point at the
 * opening quote) into out (size cap, NUL-terminated); advances *p past the
 * closing quote. Handles the small set of escapes JSON headers actually
 * use; returns -1 on malformed/unterminated input. */
static int st_parse_string(const char **p, char *out, size_t cap) {
    const char *s = *p;
    if (*s != '"') return -1;
    s++;
    size_t n = 0;
    while (*s && *s != '"') {
        char c = *s;
        if (c == '\\') {
            s++;
            switch (*s) {
                case '"': c = '"'; break; case '\\': c = '\\'; break; case '/': c = '/'; break;
                case 'n': c = '\n'; break; case 't': c = '\t'; break; case 'r': c = '\r'; break;
                default: return -1;
            }
        }
        if (n + 1 >= cap) return -1;
        out[n++] = c;
        s++;
    }
    if (*s != '"') return -1;
    out[n] = '\0';
    *p = s + 1;
    return 0;
}

/* parses a JSON array of nonnegative integers "[1,2,3]" into out (max
 * cap entries); returns count, or -1 on malformed input / overflow. */
static int st_parse_int_array(const char **p, uint64_t *out, int cap) {
    const char *s = *p;
    s += st_skip_ws(s);
    if (*s != '[') return -1;
    s++;
    int n = 0;
    for (;;) {
        s += st_skip_ws(s);
        if (*s == ']') { s++; break; }
        if (n >= cap) return -1;
        char *end;
        unsigned long long v = strtoull(s, &end, 10);
        if (end == s) return -1;
        out[n++] = (uint64_t)v;
        s = end;
        s += st_skip_ws(s);
        if (*s == ',') { s++; continue; }
        if (*s == ']') { s++; break; }
        return -1;
    }
    *p = s;
    return n;
}

/* finds the tensor's JSON object by name in the header and extracts
 * dtype string + shape + data_offsets. Returns 0 on success, -1 if the
 * name is absent or the entry is malformed. Fails closed: no partial or
 * best-guess result on error. */
static int st_find_tensor(const SafetensorsFile *st, const char *name,
                           char dtype_out[16], uint64_t shape_out[4], int *ndim_out,
                           uint64_t *off_start, uint64_t *off_end) {
    char needle[128];
    int nl = snprintf(needle, sizeof needle, "\"%s\":", name);
    const char *hit = strstr(st->header, needle);
    if (!hit) return -1;
    const char *p = hit + nl;
    p += st_skip_ws(p);
    if (*p != '{') return -1;
    p++;
    int have_dtype = 0, have_shape = 0, have_offsets = 0;
    for (;;) {
        p += st_skip_ws(p);
        if (*p == '}') { p++; break; }
        if (*p == ',') { p++; continue; }
        char key[32];
        if (st_parse_string(&p, key, sizeof key)) return -1;
        p += st_skip_ws(p);
        if (*p != ':') return -1;
        p++;
        p += st_skip_ws(p);
        if (!strcmp(key, "dtype")) {
            if (st_parse_string(&p, dtype_out, 16)) return -1;
            have_dtype = 1;
        } else if (!strcmp(key, "shape")) {
            uint64_t tmp[4];
            int nd = st_parse_int_array(&p, tmp, 4);
            if (nd < 0) return -1;
            memcpy(shape_out, tmp, sizeof(uint64_t) * (size_t)nd);
            *ndim_out = nd;
            have_shape = 1;
        } else if (!strcmp(key, "data_offsets")) {
            uint64_t tmp[2];
            if (st_parse_int_array(&p, tmp, 2) != 2) return -1;
            *off_start = tmp[0]; *off_end = tmp[1];
            have_offsets = 1;
        } else {
            return -1; /* unexpected key: fail closed rather than skip silently */
        }
    }
    return (have_dtype && have_shape && have_offsets) ? 0 : -1;
}

static int st_open(SafetensorsFile *st, const char *path) {
    memset(st, 0, sizeof *st);
    st->f = fopen(path, "rb");
    if (!st->f) return -1;
    if (fseeko(st->f, 0, SEEK_END)) { fclose(st->f); return -1; }
    off_t end = ftello(st->f);
    if (end < 8 || fseeko(st->f, 0, SEEK_SET)) { fclose(st->f); return -1; }
    st->file_size = (uint64_t)end;
    uint8_t hlen_bytes[8];
    if (fread(hlen_bytes, 1, 8, st->f) != 8) { fclose(st->f); return -1; }
    uint64_t hlen = 0;
    for (int i = 7; i >= 0; i--) hlen = (hlen << 8) | hlen_bytes[i]; /* little-endian */
    if (hlen == 0 || hlen > ST_MAX_HEADER_BYTES || hlen > st->file_size - 8) {
        fclose(st->f);
        return -1;
    }
    st->header = malloc(hlen + 1);
    if (!st->header) { fclose(st->f); return -1; }
    if (fread(st->header, 1, hlen, st->f) != hlen) { fclose(st->f); free(st->header); return -1; }
    st->header[hlen] = '\0';
    st->data_start = 8 + hlen;
    return 0;
}

static void st_close(SafetensorsFile *st) {
    if (st->f) fclose(st->f);
    free(st->header);
    memset(st, 0, sizeof *st);
}

/* reads tensor `name`, validates it is F32 with exactly the expected
 * shape (expected_ndim dims given in expected_shape), into out (must hold
 * expected element count floats). Fails closed on any mismatch. */
static int st_read_f32(SafetensorsFile *st, const char *name, const uint64_t *expected_shape, int expected_ndim, float *out) {
    char dtype[16]; uint64_t shape[4]; int ndim; uint64_t off0, off1;
    if (st_find_tensor(st, name, dtype, shape, &ndim, &off0, &off1)) {
        fprintf(stderr, "safetensors: tensor '%s' not found or malformed header entry\n", name);
        return -1;
    }
    if (strcmp(dtype, "F32")) { fprintf(stderr, "safetensors: '%s' dtype=%s, expected F32\n", name, dtype); return -1; }
    if (ndim != expected_ndim) { fprintf(stderr, "safetensors: '%s' ndim=%d, expected %d\n", name, ndim, expected_ndim); return -1; }
    for (int i = 0; i < ndim; i++) if (shape[i] != expected_shape[i]) {
        fprintf(stderr, "safetensors: '%s' shape mismatch at dim %d: got %llu expected %llu\n", name, i, (unsigned long long)shape[i], (unsigned long long)expected_shape[i]);
        return -1;
    }
    if (off1 < off0 || off0 > st->file_size - st->data_start ||
        off1 > st->file_size - st->data_start) {
        fprintf(stderr, "safetensors: '%s' data offsets are outside the file\n", name);
        return -1;
    }
    uint64_t nbytes = off1 - off0;
    if (nbytes % 4) { fprintf(stderr, "safetensors: '%s' byte length is not F32-aligned\n", name); return -1; }
    uint64_t nfloats = nbytes / 4;
    uint64_t expect_count = 1; for (int i = 0; i < ndim; i++) expect_count *= expected_shape[i];
    if (nfloats != expect_count) { fprintf(stderr, "safetensors: '%s' byte-length/shape mismatch\n", name); return -1; }
    if (fseeko(st->f, (off_t)(st->data_start + off0), SEEK_SET)) { fprintf(stderr, "safetensors: seek failed for '%s'\n", name); return -1; }
    if (fread(out, 4, nfloats, st->f) != nfloats) { fprintf(stderr, "safetensors: short read for '%s'\n", name); return -1; }
    return 0;
}

#endif
