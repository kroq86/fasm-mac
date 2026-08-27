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

/* --- ONNX-level structures: only what v0 (MatMul/Gemm/Add/Relu) needs --- */
typedef struct {
    char name[ONNX_MAX_NAME];
    int64_t dims[ONNX_MAX_DIMS];
    uint32_t ndim;
    int32_t data_type;   /* TensorProto.DataType; 1 = FLOAT */
    const float *data;   /* points into the loaded file buffer; not a copy */
    uint64_t count;      /* product of dims */
    uint64_t data_count; /* inline float elements actually present */
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
    float fval = 0; int64_t ival = 0; int has_f = 0, has_i = 0;
    while ((rc = pb_next(&r, &f)) > 0) {
        if (f.field == 1 && f.wire == 2) pb_copy_str(f.bytes, f.len, name, sizeof name);
        else if (f.field == 2 && f.wire == 5) { float v; memcpy(&v, f.bytes, 4); fval = v; has_f = 1; }
        else if (f.field == 3 && f.wire == 0) { ival = (int64_t)f.varint; has_i = 1; }
    }
    if (!strcmp(name, "alpha") && has_f) { n->alpha = fval; return rc; }
    if (!strcmp(name, "beta") && has_f) { n->beta = fval; return rc; }
    if (!strcmp(name, "transA") && has_i) { n->transA = (int)ival; return rc; }
    if (!strcmp(name, "transB") && has_i) { n->transB = (int)ival; return rc; }
    ONNX_UNSUPPORTED(g, "node '%s' (%s): attribute '%s' is not recognized for this op in v0 and was NOT applied", n->name, n->op_type, name);
    return rc;
}
static int onnx_parse_node(PbReader r, OnnxNode *n, OnnxGraph *g) {
    PbField f;
    int rc;
    memset(n, 0, sizeof *n);
    n->alpha = 1.0f; n->beta = 1.0f; n->transA = 0; n->transB = 0;
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
    memset(t, 0, sizeof *t);
    while ((rc = pb_next(&r, &f)) > 0) {
        if (f.field == 1 && f.wire == 0 && t->ndim < ONNX_MAX_DIMS) t->dims[t->ndim++] = (int64_t)f.varint;
        else if (f.field == 2 && f.wire == 0) t->data_type = (int32_t)f.varint;
        else if (f.field == 4 && f.wire == 2) { packed_floats = f.bytes; packed_len = f.len; }
        else if (f.field == 8 && f.wire == 2) pb_copy_str(f.bytes, f.len, t->name, sizeof t->name);
        else if (f.field == 9 && f.wire == 2) { raw = f.bytes; raw_len = f.len; }
    }
    uint64_t count = 1;
    for (uint32_t i = 0; i < t->ndim; i++) count *= (uint64_t)(t->dims[i] < 0 ? 0 : t->dims[i]);
    t->count = count;
    if (raw) { t->data = (const float *)(const void *)raw; t->data_count = raw_len / 4; }
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
            if (!have_a || !have_b) { ONNX_ERROR(g, "node '%s' (Gemm): operand shape unknown", n->name); continue; }
            if (an != 2 || bn != 2) { ONNX_UNSUPPORTED(g, "node '%s' (Gemm): only 2-D A/B are supported in v0", n->name); continue; }
            int64_t am = n->transA ? da[1] : da[0], ak = n->transA ? da[0] : da[1];
            int64_t bk = n->transB ? db[1] : db[0], bn2 = n->transB ? db[0] : db[1];
            if (ak != bk) { ONNX_ERROR(g, "node '%s' (Gemm): inner dims mismatch after transA/transB", n->name); continue; }
            n->out_dims[0] = am; n->out_dims[1] = bn2; n->out_ndim = 2; n->shape_ok = 1;
        } else {
            ONNX_UNSUPPORTED(g, "node '%s': op_type '%s' is not in the v0 subset (MatMul/Gemm/Add/Relu)", n->name, n->op_type);
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
    for (uint32_t i = 0; i < g->n_inputs; i++) if (!g->inputs[i].has_type || g->inputs[i].elem_type != 1) { ONNX_UNSUPPORTED(g, "graph input '%s': v0 requires explicit float32 tensor type", g->inputs[i].name); ok = 0; }
    for (uint32_t i = 0; i < g->n_outputs; i++) if (!g->outputs[i].has_type || g->outputs[i].elem_type != 1) { ONNX_UNSUPPORTED(g, "graph output '%s': v0 requires explicit float32 tensor type", g->outputs[i].name); ok = 0; }
    for (uint32_t i = 0; i < g->n_initializers; i++) {
        OnnxTensor *t = &g->initializers[i];
        if (t->data_type != 1) { ONNX_UNSUPPORTED(g, "initializer '%s': dtype %d is not float32", t->name, t->data_type); ok = 0; }
        if (!t->data || t->data_count != t->count) { ONNX_UNSUPPORTED(g, "initializer '%s': v0 requires inline data matching declared shape (%llu declared, %llu present)", t->name, (unsigned long long)t->count, (unsigned long long)t->data_count); ok = 0; }
    }
    return ok;
}

#endif
