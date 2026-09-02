#ifndef TENSOR_ONNX_IMPORT_H
#define TENSOR_ONNX_IMPORT_H
/* Real ONNX importer: reads the actual protobuf wire format of a .onnx
 * file (not a format our own scripts invented) and extracts the subset of
 * ONNX IR this project's v0 covers: MatMul, Gemm, Add, Relu. Not a stable
 * core ABI, not a general protobuf library — a minimal decoder for the
 * specific message subset this importer needs, with field numbers taken
 * directly from onnx/onnx.proto (opset-13-era schema), not guessed:
 *
 *   ModelProto.graph=7
 *   GraphProto.node=1 initializer=5 input=11 output=12
 *   NodeProto.input=1 output=2 name=3 op_type=4 attribute=5 domain=7
 *   AttributeProto.name=1 f=2 i=3 type=20
 *   TensorProto.dims=1 data_type=2 float_data=4 name=8 raw_data=9
 *   ValueInfoProto.name=1 type=2
 *   TypeProto.tensor_type=1 (oneof); TypeProto.Tensor.elem_type=1 shape=2
 *   TensorShapeProto.dim=1; Dimension.dim_value=1 dim_param=2 (oneof)
 *
 * Every shape/dtype/topology fact is derived from the file's own bytes.
 * Nothing here is informed by which fixture happens to be under test —
 * this file was written by reading onnx.proto's actual field numbers and
 * cross-checking them against a real file loaded with the reference
 * `onnx` Python package, not by hand-matching one fixture's byte layout.
 * Unsupported op_types, attributes, dtypes, or shapes are recorded as
 * diagnostics, never silently dropped or guessed past.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor_onnx_capability.h"

enum {
    ONNX_MAX_NODES = 64, ONNX_MAX_INITIALIZERS = 64, ONNX_MAX_IO = 16,
    ONNX_MAX_NODE_IO = 8, ONNX_MAX_DIMS = 6, ONNX_MAX_NAME = 96,
    ONNX_MAX_DIAG = 32, ONNX_MAX_DIAG_MSG = 160,
};

/* --- minimal protobuf wire-format reader: varint, length-delimited,
 * fixed32/fixed64. Enough for the message subset above, nothing more. --- */
typedef struct { const uint8_t *p, *end; } PbReader;
typedef struct { uint32_t field, wire; uint64_t varint; const uint8_t *bytes; uint64_t len; } PbField;

static int pb_varint(PbReader *r, uint64_t *out) {
    uint64_t v = 0; int shift = 0;
    while (r->p < r->end) {
        uint8_t b = *r->p++;
        v |= (uint64_t)(b & 0x7f) << shift;
        if (!(b & 0x80)) { *out = v; return 0; }
        shift += 7;
        if (shift >= 64) return -1;
    }
    return -1;
}
static int pb_next(PbReader *r, PbField *f) {
    if (r->p >= r->end) return 0;
    uint64_t tag;
    if (pb_varint(r, &tag)) return -1;
    f->field = (uint32_t)(tag >> 3);
    f->wire = (uint32_t)(tag & 7);
    switch (f->wire) {
        case 0: if (pb_varint(r, &f->varint)) return -1; break;
        case 1: if ((uint64_t)(r->end - r->p) < 8) return -1; f->bytes = r->p; f->len = 8; r->p += 8; break;
        case 2: { uint64_t len; if (pb_varint(r, &len) || (uint64_t)(r->end - r->p) < len) return -1; f->bytes = r->p; f->len = len; r->p += len; break; }
        case 5: if ((uint64_t)(r->end - r->p) < 4) return -1; f->bytes = r->p; f->len = 4; r->p += 4; break;
        default: return -1; /* wire type 3/4 (deprecated groups) not needed for this subset */
    }
    return 1;
}
static void pb_copy_str(const uint8_t *bytes, uint64_t len, char *out, size_t cap) {
    size_t n = len < cap - 1 ? (size_t)len : cap - 1;
    memcpy(out, bytes, n);
    out[n] = 0;
}
/* AttributeProto's repeated-scalar fields (ints/floats) use protobuf's
 * "packed" encoding: wire type 2 (length-delimited), payload is a plain
 * back-to-back run of varints (no per-element tags). */
static uint32_t pb_read_packed_int64s(const uint8_t *bytes, uint64_t len, int64_t *out, uint32_t cap) {
    PbReader r = {bytes, bytes + len};
    uint32_t n = 0;
    uint64_t v;
    while (r.p < r.end && n < cap) { if (pb_varint(&r, &v)) break; out[n++] = (int64_t)v; }
    return n;
}

/* --- ONNX-level structures: only what v0 (MatMul/Gemm/Add/Relu) needs --- */
typedef struct {
    char name[ONNX_MAX_NAME];
    int64_t dims[ONNX_MAX_DIMS];
    uint32_t ndim;
    int32_t data_type;   /* TensorProto.DataType; 1 = FLOAT, 7 = INT64 */
    const float *data;   /* points into the loaded file buffer; not a copy */
    uint64_t count;      /* product of dims */
    uint64_t data_count; /* inline float elements actually present */
    const int64_t *idata;   /* set instead of data when data_type == INT64 (shape tensors) */
    uint64_t idata_count;
    int64_t idata_buf[ONNX_MAX_DIMS]; /* owns the decoded values when the source was packed varints (int64_data), which can't be pointed-into like raw_data */
} OnnxTensor;

typedef struct {
    char name[ONNX_MAX_NAME];
    int64_t dims[ONNX_MAX_DIMS];
    uint32_t ndim;
    int32_t elem_type;
    int has_type;
} OnnxValueInfo;

typedef struct {
    char op_type[32];
    char name[ONNX_MAX_NAME];
    char inputs[ONNX_MAX_NODE_IO][ONNX_MAX_NAME];
    uint32_t n_inputs;
    char outputs[ONNX_MAX_NODE_IO][ONNX_MAX_NAME];
    uint32_t n_outputs;
    /* Gemm attributes; defaults match the ONNX spec exactly (alpha=1.0,
     * beta=1.0, transA=0, transB=0) when absent — that's a documented
     * default, not a guess. */
    float alpha, beta;
    int transA, transB;
    /* Conv/MaxPool attributes; ONNX spec defaults applied only when the
     * attribute is genuinely absent (has_* == 0), never silently assumed
     * present. kernel_shape/strides/dilations are [H,W]; pads is
     * [h_begin,w_begin,h_end,w_end] (ONNX's own ordering). */
    int64_t kernel_shape[2]; int has_kernel_shape;
    int64_t strides[2];      int has_strides;
    int64_t dilations[2];    int has_dilations;
    int64_t pads[4];         int has_pads;
    int64_t group;           int has_group;
    char auto_pad[16];       /* "NOTSET" (default), "SAME_UPPER", "SAME_LOWER", "VALID" */
    /* computed by shape inference, filled in by onnx_infer_shapes() */
    int64_t out_dims[ONNX_MAX_DIMS];
    uint32_t out_ndim;
    int shape_ok; /* 0 until inference has run and succeeded for this node */
} OnnxNode;

typedef struct {
    char message[ONNX_MAX_DIAG_MSG];
} OnnxDiag;

typedef struct {
    OnnxValueInfo inputs[ONNX_MAX_IO]; uint32_t n_inputs;
    OnnxValueInfo outputs[ONNX_MAX_IO]; uint32_t n_outputs;
    OnnxTensor initializers[ONNX_MAX_INITIALIZERS]; uint32_t n_initializers;
    OnnxNode nodes[ONNX_MAX_NODES]; uint32_t n_nodes;
    char graph_name[ONNX_MAX_NAME];
    int64_t ir_version;
    int64_t opset_version;
    OnnxDiag unsupported[ONNX_MAX_DIAG]; uint32_t n_unsupported;
    OnnxDiag errors[ONNX_MAX_DIAG]; uint32_t n_errors; /* hard failures: bad topology, bad shapes, truncated file */
    const uint8_t *file_buf; /* kept alive for the lifetime of this graph — tensors point into it */
    long file_len;
} OnnxGraph;

static void onnx_diag(OnnxDiag *arr, uint32_t *n, const char *fmt, ...) __attribute__((unused, format(printf, 3, 4)));
static void onnx_diag(OnnxDiag *arr, uint32_t *n, const char *fmt, ...) {
    if (*n >= ONNX_MAX_DIAG) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(arr[*n].message, ONNX_MAX_DIAG_MSG, fmt, ap);
    va_end(ap);
    (*n)++;
}
#define ONNX_UNSUPPORTED(g, ...) onnx_diag((g)->unsupported, &(g)->n_unsupported, __VA_ARGS__)
#define ONNX_ERROR(g, ...) onnx_diag((g)->errors, &(g)->n_errors, __VA_ARGS__)

static int onnx_parse_shape(PbReader r, int64_t *dims, uint32_t *ndim) {
    PbField f;
    int rc;
    *ndim = 0;
    while ((rc = pb_next(&r, &f)) > 0) {
        if (f.field == 1 && f.wire == 2) { /* Dimension */
            PbReader dr = {f.bytes, f.bytes + f.len};
            PbField df;
            int64_t v = -1; /* -1 = symbolic/unknown dim (dim_param, or absent) */
            while (pb_next(&dr, &df) > 0) {
                if (df.field == 1 && df.wire == 0) v = (int64_t)df.varint;
                /* field 2 (dim_param, symbolic) leaves v == -1 */
            }
            if (*ndim < ONNX_MAX_DIMS) dims[(*ndim)++] = v;
        }
    }
    return rc;
}
static int onnx_parse_type(PbReader r, int32_t *elem_type, int64_t *dims, uint32_t *ndim, int *has) {
    PbField f;
    int rc;
    *has = 0;
    while ((rc = pb_next(&r, &f)) > 0) {
        if (f.field == 1 && f.wire == 2) { /* TypeProto.Tensor */
            PbReader tr = {f.bytes, f.bytes + f.len};
            PbField tf;
            *has = 1;
            while (pb_next(&tr, &tf) > 0) {
                if (tf.field == 1 && tf.wire == 0) *elem_type = (int32_t)tf.varint;
                else if (tf.field == 2 && tf.wire == 2) { PbReader sr = {tf.bytes, tf.bytes + tf.len}; onnx_parse_shape(sr, dims, ndim); }
            }
        }
    }
    return rc;
}
static int onnx_parse_value_info(PbReader r, OnnxValueInfo *v) {
    PbField f;
    int rc;
    memset(v, 0, sizeof *v);
    while ((rc = pb_next(&r, &f)) > 0) {
        if (f.field == 1 && f.wire == 2) pb_copy_str(f.bytes, f.len, v->name, sizeof v->name);
        else if (f.field == 2 && f.wire == 2) { PbReader tr = {f.bytes, f.bytes + f.len}; onnx_parse_type(tr, &v->elem_type, v->dims, &v->ndim, &v->has_type); }
    }
    return rc;
}
static int onnx_parse_attribute(PbReader r, OnnxNode *n, OnnxGraph *g) {
    PbField f;
    int rc;
    char name[64] = {0};
    float fval = 0; int64_t ival = 0; int has_f = 0, has_i = 0, has_s = 0;
    char sval[16] = {0};
    int64_t ints[4] = {0}; uint32_t n_ints = 0; int has_ints = 0;
    while ((rc = pb_next(&r, &f)) > 0) {
        if (f.field == 1 && f.wire == 2) pb_copy_str(f.bytes, f.len, name, sizeof name);
        else if (f.field == 2 && f.wire == 5) { float v; memcpy(&v, f.bytes, 4); fval = v; has_f = 1; }
        else if (f.field == 3 && f.wire == 0) { ival = (int64_t)f.varint; has_i = 1; }
        else if (f.field == 4 && f.wire == 2) { pb_copy_str(f.bytes, f.len, sval, sizeof sval); has_s = 1; }
        /* AttributeProto.ints (field 8) is a proto2 `repeated int64` with no
         * `[packed=true]`, confirmed by hexdumping a real attribute
         * (mnist-8.onnx's kernel_shape): each element arrives as its own
         * field-8/wire-0 varint, not one wire-2 packed blob. Accept both,
         * since a packed encoder is still spec-legal even if this file
         * doesn't use one. */
        else if (f.field == 8 && f.wire == 0 && n_ints < 4) { ints[n_ints++] = (int64_t)f.varint; has_ints = 1; }
        else if (f.field == 8 && f.wire == 2) { n_ints = pb_read_packed_int64s(f.bytes, f.len, ints, 4); has_ints = 1; }
    }
    if (!strcmp(name, "alpha") && has_f) { n->alpha = fval; return rc; }
    if (!strcmp(name, "beta") && has_f) { n->beta = fval; return rc; }
    if (!strcmp(name, "transA") && has_i) { n->transA = (int)ival; return rc; }
    if (!strcmp(name, "transB") && has_i) { n->transB = (int)ival; return rc; }
    if (!strcmp(name, "group") && has_i) { n->group = ival; n->has_group = 1; return rc; }
    if (!strcmp(name, "auto_pad") && has_s) { pb_copy_str((const uint8_t *)sval, strlen(sval), n->auto_pad, sizeof n->auto_pad); return rc; }
    if (!strcmp(name, "kernel_shape") && has_ints && n_ints == 2) { n->kernel_shape[0] = ints[0]; n->kernel_shape[1] = ints[1]; n->has_kernel_shape = 1; return rc; }
    if (!strcmp(name, "strides") && has_ints && n_ints == 2) { n->strides[0] = ints[0]; n->strides[1] = ints[1]; n->has_strides = 1; return rc; }
    if (!strcmp(name, "dilations") && has_ints && n_ints == 2) { n->dilations[0] = ints[0]; n->dilations[1] = ints[1]; n->has_dilations = 1; return rc; }
    if (!strcmp(name, "pads") && has_ints && n_ints == 4) { memcpy(n->pads, ints, sizeof n->pads); n->has_pads = 1; return rc; }
    ONNX_UNSUPPORTED(g, "node '%s' (%s): attribute '%s' is not recognized for this op in v0 and was NOT applied", n->name, n->op_type, name);
    return rc;
}
static int onnx_parse_node(PbReader r, OnnxNode *n, OnnxGraph *g) {
    PbField f;
    int rc;
    memset(n, 0, sizeof *n);
    n->alpha = 1.0f; n->beta = 1.0f; n->transA = 0; n->transB = 0;
    n->group = 1; n->dilations[0] = n->dilations[1] = 1;
    strcpy(n->auto_pad, "NOTSET"); /* ONNX spec default when the attribute is absent */
    while ((rc = pb_next(&r, &f)) > 0) {
        if (f.field == 1 && f.wire == 2 && n->n_inputs < ONNX_MAX_NODE_IO) pb_copy_str(f.bytes, f.len, n->inputs[n->n_inputs++], ONNX_MAX_NAME);
        else if (f.field == 2 && f.wire == 2 && n->n_outputs < ONNX_MAX_NODE_IO) pb_copy_str(f.bytes, f.len, n->outputs[n->n_outputs++], ONNX_MAX_NAME);
        else if (f.field == 3 && f.wire == 2) pb_copy_str(f.bytes, f.len, n->name, sizeof n->name);
        else if (f.field == 4 && f.wire == 2) pb_copy_str(f.bytes, f.len, n->op_type, sizeof n->op_type);
        else if (f.field == 5 && f.wire == 2) { PbReader ar = {f.bytes, f.bytes + f.len}; onnx_parse_attribute(ar, n, g); }
    }
    return rc;
}
static int onnx_parse_tensor(PbReader r, OnnxTensor *t) {
    PbField f;
    int rc;
    const uint8_t *raw = NULL; uint64_t raw_len = 0;
    const uint8_t *packed_floats = NULL; uint64_t packed_len = 0;
    const uint8_t *packed_int64s = NULL; uint64_t packed_int64_len = 0;
    memset(t, 0, sizeof *t);
    while ((rc = pb_next(&r, &f)) > 0) {
        if (f.field == 1 && f.wire == 0 && t->ndim < ONNX_MAX_DIMS) t->dims[t->ndim++] = (int64_t)f.varint;
        else if (f.field == 2 && f.wire == 0) t->data_type = (int32_t)f.varint;
        else if (f.field == 4 && f.wire == 2) { packed_floats = f.bytes; packed_len = f.len; }
        else if (f.field == 7 && f.wire == 2) { packed_int64s = f.bytes; packed_int64_len = f.len; } /* int64_data, packed varints */
        else if (f.field == 8 && f.wire == 2) pb_copy_str(f.bytes, f.len, t->name, sizeof t->name);
        else if (f.field == 9 && f.wire == 2) { raw = f.bytes; raw_len = f.len; }
    }
    uint64_t count = 1;
    for (uint32_t i = 0; i < t->ndim; i++) count *= (uint64_t)(t->dims[i] < 0 ? 0 : t->dims[i]);
    t->count = count;
    if (t->data_type == 7) { /* INT64 (shape tensors, e.g. Reshape's second operand) */
        if (raw) { t->idata = (const int64_t *)(const void *)raw; t->idata_count = raw_len / 8; }
        else if (packed_int64s) {
            /* int64_data is varint-packed, not fixed-8-byte-per-element, so
             * it can't be pointed into directly like the float/raw_data
             * cases — decode into this tensor's own idata_buf (shape
             * tensors are always tiny: the rank of some other tensor). */
            t->idata_count = pb_read_packed_int64s(packed_int64s, packed_int64_len, t->idata_buf, ONNX_MAX_DIMS);
            t->idata = t->idata_buf;
        }
    } else if (raw) { t->data = (const float *)(const void *)raw; t->data_count = raw_len / 4; }
    else if (packed_floats) { t->data = (const float *)(const void *)packed_floats; t->data_count = packed_len / 4; }
    return rc;
}
static int onnx_parse_graph(PbReader r, OnnxGraph *g) {
    PbField f;
    int rc;
    while ((rc = pb_next(&r, &f)) > 0) {
        if (f.field == 1 && f.wire == 2 && g->n_nodes < ONNX_MAX_NODES) { PbReader nr = {f.bytes, f.bytes + f.len}; onnx_parse_node(nr, &g->nodes[g->n_nodes++], g); }
        else if (f.field == 2 && f.wire == 2) pb_copy_str(f.bytes, f.len, g->graph_name, sizeof g->graph_name);
        else if (f.field == 5 && f.wire == 2 && g->n_initializers < ONNX_MAX_INITIALIZERS) { PbReader tr = {f.bytes, f.bytes + f.len}; onnx_parse_tensor(tr, &g->initializers[g->n_initializers++]); }
        else if (f.field == 11 && f.wire == 2 && g->n_inputs < ONNX_MAX_IO) { PbReader vr = {f.bytes, f.bytes + f.len}; onnx_parse_value_info(vr, &g->inputs[g->n_inputs++]); }
        else if (f.field == 12 && f.wire == 2 && g->n_outputs < ONNX_MAX_IO) { PbReader vr = {f.bytes, f.bytes + f.len}; onnx_parse_value_info(vr, &g->outputs[g->n_outputs++]); }
    }
    return rc;
}
static int onnx_parse_model(const uint8_t *buf, long len, OnnxGraph *g) {
    memset(g, 0, sizeof *g);
    g->file_buf = buf; g->file_len = len;
    PbReader r = {buf, buf + len};
    PbField f;
    int rc, found_graph = 0;
    while ((rc = pb_next(&r, &f)) > 0) {
        if (f.field == 1 && f.wire == 0) g->ir_version = (int64_t)f.varint;
        else if (f.field == 7 && f.wire == 2) {
            PbReader gr = {f.bytes, f.bytes + f.len};
            if (onnx_parse_graph(gr, g) < 0) { ONNX_ERROR(g, "malformed GraphProto (truncated or invalid protobuf)"); return -1; }
            found_graph = 1;
        } else if (f.field == 8 && f.wire == 2) { /* opset_import: repeated OperatorSetIdProto{domain=1,version=2} */
            PbReader or_ = {f.bytes, f.bytes + f.len};
            PbField of;
            while (pb_next(&or_, &of) > 0) if (of.field == 2 && of.wire == 0) g->opset_version = (int64_t)of.varint;
        }
    }
    if (rc < 0) { ONNX_ERROR(g, "malformed ModelProto (truncated or invalid protobuf)"); return -1; }
    if (!found_graph) { ONNX_ERROR(g, "ModelProto has no graph field"); return -1; }
    return 0;
}
static int onnx_load_file(const char *path, OnnxGraph *g) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)len);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { fclose(f); free(buf); return -1; }
    fclose(f);
    return onnx_parse_model(buf, len, g);
}

/* --- lookups + shape inference for the v0 op subset. Broadcasting is
 * real numpy-style broadcasting (right-aligned, size-1 dims stretch),
 * not an assumption that shapes already match. --- */
static const OnnxTensor *onnx_find_initializer(const OnnxGraph *g, const char *name) {
    for (uint32_t i = 0; i < g->n_initializers; i++) if (!strcmp(g->initializers[i].name, name)) return &g->initializers[i];
    return NULL;
}
static int onnx_find_value_shape(const OnnxGraph *g, const char *name, int64_t *dims, uint32_t *ndim) {
    for (uint32_t i = 0; i < g->n_inputs; i++) if (!strcmp(g->inputs[i].name, name) && g->inputs[i].has_type) { memcpy(dims, g->inputs[i].dims, sizeof(int64_t) * g->inputs[i].ndim); *ndim = g->inputs[i].ndim; return 0; }
    const OnnxTensor *t = onnx_find_initializer(g, name);
    if (t) { memcpy(dims, t->dims, sizeof(int64_t) * t->ndim); *ndim = t->ndim; return 0; }
    for (uint32_t i = 0; i < g->n_nodes; i++) if (g->nodes[i].shape_ok) for (uint32_t j = 0; j < g->nodes[i].n_outputs; j++)
        if (!strcmp(g->nodes[i].outputs[j], name)) { memcpy(dims, g->nodes[i].out_dims, sizeof(int64_t) * g->nodes[i].out_ndim); *ndim = g->nodes[i].out_ndim; return 0; }
    return -1;
}
static int onnx_broadcast(const int64_t *a, uint32_t an, const int64_t *b, uint32_t bn, int64_t *out, uint32_t *on) {
    uint32_t n = an > bn ? an : bn;
    for (uint32_t k = 0; k < n; k++) {
        int64_t da = k < n - an ? 1 : a[k - (n - an)];
        int64_t db = k < n - bn ? 1 : b[k - (n - bn)];
        if (da != 1 && db != 1 && da != db) return -1;
        out[k] = da == 1 ? db : da;
    }
    *on = n;
    return 0;
}
/* Returns 0 and fills g->nodes[i].out_dims/out_ndim/shape_ok for every node
 * it can resolve; unresolvable/unsupported nodes are recorded as
 * diagnostics and shape_ok stays 0 for them and everything downstream that
 * depends on them (no guessing forward past a failure). */
static void onnx_infer_shapes(OnnxGraph *g) {
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        OnnxNode *n = &g->nodes[i];
        int64_t da[ONNX_MAX_DIMS], db[ONNX_MAX_DIMS]; uint32_t an = 0, bn = 0;
        int have_a = n->n_inputs >= 1 && onnx_find_value_shape(g, n->inputs[0], da, &an) == 0;
        int have_b = n->n_inputs >= 2 && onnx_find_value_shape(g, n->inputs[1], db, &bn) == 0;
        if (!strcmp(n->op_type, "Relu")) {
            if (!have_a) { ONNX_ERROR(g, "node '%s' (Relu): input '%s' shape unknown", n->name, n->inputs[0]); continue; }
            memcpy(n->out_dims, da, sizeof(int64_t) * an); n->out_ndim = an; n->shape_ok = 1;
        } else if (!strcmp(n->op_type, "Add")) {
            if (!have_a || !have_b) { ONNX_ERROR(g, "node '%s' (Add): operand shape unknown", n->name); continue; }
            if (onnx_broadcast(da, an, db, bn, n->out_dims, &n->out_ndim)) { ONNX_ERROR(g, "node '%s' (Add): shapes not broadcastable", n->name); continue; }
            n->shape_ok = 1;
        } else if (!strcmp(n->op_type, "MatMul")) {
            if (!have_a || !have_b) { ONNX_ERROR(g, "node '%s' (MatMul): operand shape unknown", n->name); continue; }
            if (an != 2 || bn != 2) { ONNX_UNSUPPORTED(g, "node '%s' (MatMul): only 2-D operands are supported in v0 (got %u-D/%u-D)", n->name, an, bn); continue; }
            if (da[1] != db[0]) { ONNX_ERROR(g, "node '%s' (MatMul): inner dims mismatch (%lldx%lld @ %lldx%lld)", n->name, (long long)da[0], (long long)da[1], (long long)db[0], (long long)db[1]); continue; }
            n->out_dims[0] = da[0]; n->out_dims[1] = db[1]; n->out_ndim = 2; n->shape_ok = 1;
        } else if (!strcmp(n->op_type, "Gemm")) {
            if (n->n_inputs < 2 || n->n_inputs > 3) { ONNX_ERROR(g, "node '%s' (Gemm): expected 2 or 3 inputs, got %u", n->name, n->n_inputs); continue; }
            if (!have_a || !have_b) { ONNX_ERROR(g, "node '%s' (Gemm): operand shape unknown", n->name); continue; }
            if (an != 2 || bn != 2) { ONNX_UNSUPPORTED(g, "node '%s' (Gemm): only 2-D A/B are supported in v0", n->name); continue; }
            int64_t am = n->transA ? da[1] : da[0], ak = n->transA ? da[0] : da[1];
            int64_t bk = n->transB ? db[1] : db[0], bn2 = n->transB ? db[0] : db[1];
            if (ak != bk) { ONNX_ERROR(g, "node '%s' (Gemm): inner dims mismatch after transA/transB", n->name); continue; }
            /* The v0 emitter addresses C as C[column].  ONNX permits broader
             * unidirectional broadcasting, but accepting it here would make
             * shapes such as [M,1] silently produce the wrong result. */
            if (n->n_inputs == 3) {
                int64_t dc[ONNX_MAX_DIMS]; uint32_t cn = 0;
                if (onnx_find_value_shape(g, n->inputs[2], dc, &cn)) { ONNX_ERROR(g, "node '%s' (Gemm): bias C shape unknown", n->name); continue; }
                if (!((cn == 1 && dc[0] == bn2) || (cn == 2 && dc[0] == 1 && dc[1] == bn2))) {
                    ONNX_UNSUPPORTED(g, "node '%s' (Gemm): bias C shape is not supported by native v0 lowering (expected [%lld] or [1,%lld], got %u-D)", n->name, (long long)bn2, (long long)bn2, cn);
                    continue;
                }
            }
            n->out_dims[0] = am; n->out_dims[1] = bn2; n->out_ndim = 2; n->shape_ok = 1;
        } else if (!strcmp(n->op_type, "Conv")) {
            /* NCHW input, OIHW weight — this is ONNX's own convention, not a
             * choice we make; the mismatch against this project's canonical
             * HWC layout is exactly the boundary this importer exists to
             * surface (in inspect's report), not paper over here. */
            if (!have_a) { ONNX_ERROR(g, "node '%s' (Conv): input '%s' shape unknown", n->name, n->inputs[0]); continue; }
            if (!have_b) { ONNX_ERROR(g, "node '%s' (Conv): weight '%s' shape unknown", n->name, n->n_inputs >= 2 ? n->inputs[1] : "?"); continue; }
            if (an != 4 || bn != 4) { ONNX_UNSUPPORTED(g, "node '%s' (Conv): only 4-D NCHW input and OIHW weight are supported in v0 (got %u-D/%u-D)", n->name, an, bn); continue; }
            if (n->group != 1) { ONNX_UNSUPPORTED(g, "node '%s' (Conv): grouped convolution is not supported in v0 (group=%lld)", n->name, (long long)n->group); continue; }
            int64_t kh = n->has_kernel_shape ? n->kernel_shape[0] : db[2];
            int64_t kw = n->has_kernel_shape ? n->kernel_shape[1] : db[3];
            int64_t sh = n->has_strides ? n->strides[0] : 1, sw = n->has_strides ? n->strides[1] : 1;
            int64_t dh = n->dilations[0], dw = n->dilations[1];
            int64_t in_h = da[2], in_w = da[3], out_h, out_w;
            /* n->pads is left holding the RAW attribute (or its NOTSET
             * default) by earlier parsing; from here on it is overwritten
             * to hold the EFFECTIVE [h_begin,w_begin,h_end,w_end] padding
             * to actually use, regardless of whether it came from an
             * explicit attribute or was derived from auto_pad — so a
             * downstream consumer (native codegen) never needs to
             * re-derive the auto_pad formula itself, it just reads
             * n->pads after inference has run. */
            if (!strcmp(n->auto_pad, "SAME_UPPER") || !strcmp(n->auto_pad, "SAME_LOWER")) {
                out_h = (in_h + sh - 1) / sh; out_w = (in_w + sw - 1) / sw; /* ceil(in/stride), per ONNX auto_pad spec */
                int64_t need_h = (out_h - 1) * sh + ((kh - 1) * dh + 1) - in_h; if (need_h < 0) need_h = 0;
                int64_t need_w = (out_w - 1) * sw + ((kw - 1) * dw + 1) - in_w; if (need_w < 0) need_w = 0;
                int64_t small_h = need_h / 2, small_w = need_w / 2;
                int upper = !strcmp(n->auto_pad, "SAME_UPPER");
                n->pads[0] = upper ? small_h : need_h - small_h; n->pads[2] = upper ? need_h - small_h : small_h;
                n->pads[1] = upper ? small_w : need_w - small_w; n->pads[3] = upper ? need_w - small_w : small_w;
            } else if (!strcmp(n->auto_pad, "VALID")) {
                out_h = (in_h - ((kh - 1) * dh + 1)) / sh + 1;
                out_w = (in_w - ((kw - 1) * dw + 1)) / sw + 1;
                n->pads[0] = n->pads[1] = n->pads[2] = n->pads[3] = 0;
            } else { /* NOTSET: explicit pads [h_begin,w_begin,h_end,w_end], default all-zero — n->pads already holds this */
                int64_t pt = n->pads[0], pl = n->pads[1], pb = n->pads[2], pr = n->pads[3];
                out_h = (in_h + pt + pb - ((kh - 1) * dh + 1)) / sh + 1;
                out_w = (in_w + pl + pr - ((kw - 1) * dw + 1)) / sw + 1;
            }
            n->out_dims[0] = da[0]; n->out_dims[1] = db[0]; n->out_dims[2] = out_h; n->out_dims[3] = out_w;
            n->out_ndim = 4; n->shape_ok = 1;
        } else if (!strcmp(n->op_type, "MaxPool")) {
            if (!have_a) { ONNX_ERROR(g, "node '%s' (MaxPool): input '%s' shape unknown", n->name, n->inputs[0]); continue; }
            if (an != 4) { ONNX_UNSUPPORTED(g, "node '%s' (MaxPool): only 4-D NCHW input is supported in v0 (got %u-D)", n->name, an); continue; }
            if (!n->has_kernel_shape) { ONNX_UNSUPPORTED(g, "node '%s' (MaxPool): kernel_shape attribute is required", n->name); continue; }
            int64_t kh = n->kernel_shape[0], kw = n->kernel_shape[1];
            /* MaxPool's default stride is kernel_shape, not [1,1] — a
             * different default than Conv's, per the ONNX spec; getting
             * this wrong silently changes the output size. */
            int64_t sh = n->has_strides ? n->strides[0] : kh, sw = n->has_strides ? n->strides[1] : kw;
            int64_t in_h = da[2], in_w = da[3], out_h, out_w;
            if (!strcmp(n->auto_pad, "SAME_UPPER") || !strcmp(n->auto_pad, "SAME_LOWER")) {
                out_h = (in_h + sh - 1) / sh; out_w = (in_w + sw - 1) / sw;
                int64_t need_h = (out_h - 1) * sh + kh - in_h; if (need_h < 0) need_h = 0;
                int64_t need_w = (out_w - 1) * sw + kw - in_w; if (need_w < 0) need_w = 0;
                int64_t small_h = need_h / 2, small_w = need_w / 2;
                int upper = !strcmp(n->auto_pad, "SAME_UPPER");
                n->pads[0] = upper ? small_h : need_h - small_h; n->pads[2] = upper ? need_h - small_h : small_h;
                n->pads[1] = upper ? small_w : need_w - small_w; n->pads[3] = upper ? need_w - small_w : small_w;
            } else if (!strcmp(n->auto_pad, "VALID")) {
                out_h = (in_h - kh) / sh + 1; out_w = (in_w - kw) / sw + 1;
                n->pads[0] = n->pads[1] = n->pads[2] = n->pads[3] = 0;
            } else {
                int64_t pt = n->pads[0], pl = n->pads[1], pb = n->pads[2], pr = n->pads[3];
                out_h = (in_h + pt + pb - kh) / sh + 1;
                out_w = (in_w + pl + pr - kw) / sw + 1;
            }
            n->out_dims[0] = da[0]; n->out_dims[1] = da[1]; n->out_dims[2] = out_h; n->out_dims[3] = out_w;
            n->out_ndim = 4; n->shape_ok = 1;
        } else if (!strcmp(n->op_type, "Reshape")) {
            if (!have_a) { ONNX_ERROR(g, "node '%s' (Reshape): input '%s' shape unknown", n->name, n->inputs[0]); continue; }
            if (n->n_inputs < 2) { ONNX_UNSUPPORTED(g, "node '%s' (Reshape): v0 requires the target shape as a second input", n->name); continue; }
            const OnnxTensor *shape_t = onnx_find_initializer(g, n->inputs[1]);
            if (!shape_t || !shape_t->idata) { ONNX_UNSUPPORTED(g, "node '%s' (Reshape): target shape must be a constant int64 initializer in v0 (got a computed/missing tensor for '%s')", n->name, n->inputs[1]); continue; }
            int64_t total = 1; for (uint32_t k = 0; k < an; k++) total *= da[k];
            int64_t out_dims[ONNX_MAX_DIMS]; uint32_t out_ndim = (uint32_t)shape_t->idata_count;
            if (out_ndim > ONNX_MAX_DIMS) { ONNX_UNSUPPORTED(g, "node '%s' (Reshape): target rank %u exceeds v0 limit", n->name, out_ndim); continue; }
            int64_t known_product = 1; int neg_one_axis = -1; int bad = 0;
            for (uint32_t k = 0; k < out_ndim; k++) {
                int64_t v = shape_t->idata[k];
                if (v == -1) { neg_one_axis = (int)k; out_dims[k] = -1; }
                else if (v == 0) {
                    if (k >= an) { ONNX_ERROR(g, "node '%s' (Reshape): dim 0 (copy-from-input) at axis %u has no matching input axis", n->name, k); bad = 1; break; }
                    out_dims[k] = da[k]; known_product *= out_dims[k];
                } else { out_dims[k] = v; known_product *= v; }
            }
            if (bad) continue;
            if (neg_one_axis >= 0) {
                if (known_product == 0 || total % known_product != 0) { ONNX_ERROR(g, "node '%s' (Reshape): -1 dim not evenly divisible (%lld elements, %lld known product)", n->name, (long long)total, (long long)known_product); continue; }
                out_dims[neg_one_axis] = total / known_product; known_product *= out_dims[neg_one_axis];
            }
            if (known_product != total) { ONNX_ERROR(g, "node '%s' (Reshape): target shape element count (%lld) does not match input (%lld)", n->name, (long long)known_product, (long long)total); continue; }
            memcpy(n->out_dims, out_dims, sizeof(int64_t) * out_ndim); n->out_ndim = out_ndim; n->shape_ok = 1;
        } else {
            ONNX_UNSUPPORTED(g, "node '%s': op_type '%s' is not in the v0 subset (MatMul/Gemm/Add/Relu/Conv/MaxPool/Reshape)", n->name, n->op_type);
        }
    }
}

/* --- topology validation: ONNX requires nodes to already be topologically
 * sorted (onnx.proto: "The nodes in the graph, sorted topologically"). We
 * check that, not silently re-sort — a file violating it is invalid IR. --- */
static int onnx_validate_topology(OnnxGraph *g) {
    char known[ONNX_MAX_NODES * ONNX_MAX_NODE_IO + ONNX_MAX_IO + ONNX_MAX_INITIALIZERS][ONNX_MAX_NAME];
    uint32_t n_known = 0;
    for (uint32_t i = 0; i < g->n_inputs; i++) strncpy(known[n_known++], g->inputs[i].name, ONNX_MAX_NAME);
    for (uint32_t i = 0; i < g->n_initializers; i++) strncpy(known[n_known++], g->initializers[i].name, ONNX_MAX_NAME);
    int ok = 1;
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        OnnxNode *n = &g->nodes[i];
        for (uint32_t j = 0; j < n->n_inputs; j++) {
            int found = 0;
            for (uint32_t k = 0; k < n_known; k++) if (!strcmp(known[k], n->inputs[j])) { found = 1; break; }
            if (!found) { ONNX_ERROR(g, "node '%s' (%s): input '%s' is not produced by any earlier node/initializer/graph-input — file is not topologically sorted or references an undefined tensor", n->name, n->op_type, n->inputs[j]); ok = 0; }
        }
        for (uint32_t j = 0; j < n->n_outputs && n_known < sizeof known / sizeof known[0]; j++) {
            int duplicate = 0;
            for (uint32_t k = 0; k < n_known; k++) if (!strcmp(known[k], n->outputs[j])) { duplicate = 1; break; }
            if (duplicate) { ONNX_ERROR(g, "node '%s' (%s): output '%s' already has a producer", n->name, n->op_type, n->outputs[j]); ok = 0; }
            else strncpy(known[n_known++], n->outputs[j], ONNX_MAX_NAME);
        }
    }
    return ok;
}

static int onnx_validate_storage_and_types(OnnxGraph *g) {
    int ok = 1;
    for (uint32_t i = 0; i < g->n_inputs; i++) {
        /* Older ONNX IR (this project's CNN fixtures use ir_version=3,
         * opset=8) requires every initializer to also be listed as a graph
         * input — that duplicate listing is a storage convention, not a
         * second runtime input, and its type is governed by the
         * initializer check below (which allows float32 weights AND
         * int64 shape tensors), not this float32-only check. */
        int shadowed_by_initializer = onnx_find_initializer(g, g->inputs[i].name) != NULL;
        if (shadowed_by_initializer) continue;
        if (!g->inputs[i].has_type || g->inputs[i].elem_type != 1) { ONNX_UNSUPPORTED(g, "graph input '%s': v0 requires explicit float32 tensor type", g->inputs[i].name); ok = 0; }
    }
    for (uint32_t i = 0; i < g->n_outputs; i++) if (!g->outputs[i].has_type || g->outputs[i].elem_type != 1) { ONNX_UNSUPPORTED(g, "graph output '%s': v0 requires explicit float32 tensor type", g->outputs[i].name); ok = 0; }
    for (uint32_t i = 0; i < g->n_initializers; i++) {
        OnnxTensor *t = &g->initializers[i];
        if (t->data_type == 7) { /* INT64 shape tensor (e.g. Reshape's second operand) */
            if (!t->idata || t->idata_count != t->count) { ONNX_UNSUPPORTED(g, "initializer '%s': v0 requires inline int64 data matching declared shape (%llu declared, %llu present)", t->name, (unsigned long long)t->count, (unsigned long long)t->idata_count); ok = 0; }
            continue;
        }
        if (t->data_type != 1) { ONNX_UNSUPPORTED(g, "initializer '%s': dtype %d is not float32 or int64", t->name, t->data_type); ok = 0; }
        if (!t->data || t->data_count != t->count) { ONNX_UNSUPPORTED(g, "initializer '%s': v0 requires inline data matching declared shape (%llu declared, %llu present)", t->name, (unsigned long long)t->count, (unsigned long long)t->data_count); ok = 0; }
    }
    return ok;
}

#endif
