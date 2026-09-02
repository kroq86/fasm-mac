/* tensorctl inspect: reads a real external .onnx file (the actual
 * protobuf wire format, via tensor_onnx_import.h — not a format any script
 * in this repo invented) and reports what it can determine about it,
 * tagged by provenance so a reader can tell a fact read from the file
 * apart from a number this tool computed apart from a thing it simply
 * doesn't know without running something:
 *
 *   [model]       read directly from the file's bytes
 *   [derived]     computed from the file via static graph analysis
 *   [estimated]   a static cost model, not a measurement
 *   [measured]    would require actually running something; inspect never
 *                 does, so this is always "none" here
 *   [unknown]     genuinely not knowable from the file alone (e.g. how it
 *                 compares to another engine)
 *   [unsupported] anything in the file this importer's v0 subset
 *                 (MatMul/Gemm/Add/Relu/Conv/MaxPool/Reshape) can't
 *                 interpret — reported explicitly, never silently
 *                 dropped or guessed past
 *
 * v0 subset: MatMul/Gemm/Add/Relu/Conv/MaxPool/Reshape. Conv/MaxPool were
 * added after the plain-MLP path (Wine) was proven end to end, per the
 * explicit ordering this was asked for — and specifically to force this
 * importer to reason about NCHW (ONNX Conv/MaxPool convention) vs this
 * project's own canonical HWC layout, not just to check off more op
 * support. The '[derived] layout' section below is the actual point of
 * that work: it states what transform is required and its cost, verified
 * directly against tensor_semantic_compiler.h's k_conv_fwd indexing, not
 * assumed or guessed.
 */
#include "tensor_onnx_import.h"

static uint64_t onnx_elem_count(const int64_t *dims, uint32_t ndim) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < ndim; i++) n *= (uint64_t)(dims[i] < 0 ? 1 : dims[i]); /* unknown dims counted as 1: a floor, not a guess presented as fact */
    return n;
}
static void print_shape(const int64_t *dims, uint32_t ndim) {
    printf("[");
    for (uint32_t i = 0; i < ndim; i++) { if (dims[i] < 0) printf("?"); else printf("%lld", (long long)dims[i]); if (i + 1 < ndim) printf(","); }
    printf("]");
}
static const char *dtype_name(int32_t dt) { return dt == 1 ? "float32" : dt == 7 ? "int64" : dt == 0 ? "undefined" : "other"; }

/* Same MB/MBR adjacency rule the canonical compiler's fusion pass already
 * uses (tensor_semantic_compiler.h's detect_fusion): a MatMul immediately
 * followed by an Add that consumes ONLY that MatMul's output, optionally
 * followed by a Relu that likewise has exactly one consumer. Reported
 * here as a static graph property, not executed. */
static uint32_t onnx_consumers_of(const OnnxGraph *g, const char *tensor_name) {
    uint32_t c = 0;
    for (uint32_t i = 0; i < g->n_nodes; i++) for (uint32_t j = 0; j < g->nodes[i].n_inputs; j++) if (!strcmp(g->nodes[i].inputs[j], tensor_name)) c++;
    for (uint32_t i = 0; i < g->n_outputs; i++) if (!strcmp(g->outputs[i].name, tensor_name)) c++; /* graph output counts as an external consumer */
    return c;
}

int run_inspect(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: tensorctl inspect MODEL.onnx\n"); return 2; }
    OnnxGraph g;
    if (onnx_load_file(argv[1], &g)) { fprintf(stderr, "tensorctl inspect: could not read '%s' (missing file, or not a readable ONNX protobuf)\n", argv[1]); return 3; }
    if (g.n_errors) {
        for (uint32_t i = 0; i < g.n_errors; i++) fprintf(stderr, "[error] %s\n", g.errors[i].message);
        return 3;
    }
    onnx_validate_storage_and_types(&g);
    if (!onnx_validate_topology(&g)) {
        for (uint32_t i = 0; i < g.n_errors; i++) fprintf(stderr, "[error] %s\n", g.errors[i].message);
        return 3;
    }
    onnx_infer_shapes(&g);
    if (g.n_errors) {
        for (uint32_t i = 0; i < g.n_errors; i++) fprintf(stderr, "[error] %s\n", g.errors[i].message);
        return 3;
    }

    printf("[model] file: %s (ir_version=%lld opset=%lld graph='%s')\n", argv[1], (long long)g.ir_version, (long long)g.opset_version, g.graph_name);
    for (uint32_t i = 0; i < g.n_inputs; i++) { printf("[model] input: %s ", g.inputs[i].name); print_shape(g.inputs[i].dims, g.inputs[i].ndim); printf(" %s\n", g.inputs[i].has_type ? dtype_name(g.inputs[i].elem_type) : "dtype-unspecified"); }
    for (uint32_t i = 0; i < g.n_outputs; i++) { printf("[model] output: %s ", g.outputs[i].name); print_shape(g.outputs[i].dims, g.outputs[i].ndim); printf(" %s\n", g.outputs[i].has_type ? dtype_name(g.outputs[i].elem_type) : "dtype-unspecified"); }
    printf("[model] nodes:");
    for (uint32_t i = 0; i < g.n_nodes; i++) printf(" %s", g.nodes[i].op_type);
    printf("\n");
    printf("[model] initializers:");
    for (uint32_t i = 0; i < g.n_initializers; i++) printf(" %s%s", g.initializers[i].name, i + 1 < g.n_initializers ? "," : "");
    printf("\n");

    /* [derived] parameters */
    uint64_t total_params = 0;
    for (uint32_t i = 0; i < g.n_initializers; i++) {
        printf("[derived] parameter %s: ", g.initializers[i].name);
        print_shape(g.initializers[i].dims, g.initializers[i].ndim);
        printf(" %s (%llu elements)\n", dtype_name(g.initializers[i].data_type), (unsigned long long)g.initializers[i].count);
        total_params += g.initializers[i].count;
    }
    printf("[derived] total parameters: %llu (%llu bytes at float32)\n", (unsigned long long)total_params, (unsigned long long)total_params * 4);

    /* [derived] activations, per resolved node */
    uint64_t peak_activation_single = 0;
    for (uint32_t i = 0; i < g.n_nodes; i++) {
        OnnxNode *n = &g.nodes[i];
        if (!n->shape_ok) continue;
        uint64_t elems = onnx_elem_count(n->out_dims, n->out_ndim);
        printf("[derived] activation %s (%s output): ", n->outputs[0], n->op_type);
        print_shape(n->out_dims, n->out_ndim);
        printf(" (%llu elements, %llu bytes)\n", (unsigned long long)elems, (unsigned long long)elems * 4);
        if (elems > peak_activation_single) peak_activation_single = elems;
    }

    /* [derived] fusion: same MB/MBR adjacency rule as the canonical compiler's fusion pass */
    unsigned fused_groups = 0;
    for (uint32_t i = 0; i < g.n_nodes; i++) {
        OnnxNode *mm = &g.nodes[i];
        if (strcmp(mm->op_type, "MatMul") && strcmp(mm->op_type, "Gemm")) continue;
        if (i + 1 >= g.n_nodes) continue;
        OnnxNode *add = &g.nodes[i + 1];
        if (strcmp(add->op_type, "Add") || add->n_inputs < 1 || strcmp(add->inputs[0], mm->outputs[0])) continue;
        if (onnx_consumers_of(&g, mm->outputs[0]) != 1) continue;
        int has_relu = i + 2 < g.n_nodes && !strcmp(g.nodes[i + 2].op_type, "Relu") && g.nodes[i + 2].n_inputs >= 1 && !strcmp(g.nodes[i + 2].inputs[0], add->outputs[0]) && onnx_consumers_of(&g, add->outputs[0]) == 1;
        printf("[derived] fusion: %s('%s')+Add('%s')%s can fuse (matmul+bias%s pattern, consumer-count==1) — same rule the canonical compiler's fusion pass uses\n",
               mm->op_type, mm->name, add->name, has_relu ? "+Relu" : "", has_relu ? "+relu" : "");
        fused_groups++;
    }
    if (!fused_groups) printf("[derived] fusion: no matmul+bias[+relu] groups found\n");

    /* [derived] layout: NCHW (ONNX Conv/MaxPool convention) vs this
     * project's canonical HWC (CONV/POOL in tensor_semantic_compiler.h).
     * Verified directly against k_conv_fwd's own indexing —
     * activation: a->tensor.data[(ih*WIN+iw)*CIN+ci]              -> HWC
     * weight:     b->tensor.data[co*(CIN*KH*KW)+(ci*KH+kh)*KW+kw] -> OIHW
     * — so the weight layout already matches ONNX's Conv weight (also
     * OIHW) with no transform; only the activation needs a transpose.
     * This is the actual boundary this importer exists to surface, not
     * just another op to check off. */
    int any_conv_or_pool = 0;
    for (uint32_t i = 0; i < g.n_nodes; i++) {
        OnnxNode *n = &g.nodes[i];
        int is_conv = !strcmp(n->op_type, "Conv");
        int is_pool = !strcmp(n->op_type, "MaxPool");
        if (!is_conv && !is_pool) continue;
        any_conv_or_pool = 1;
        int64_t in_dims[ONNX_MAX_DIMS]; uint32_t in_ndim;
        if (n->n_inputs < 1 || onnx_find_value_shape(&g, n->inputs[0], in_dims, &in_ndim) != 0 || in_ndim != 4) {
            printf("[derived] layout %s('%s'): input shape unavailable or not 4-D — cannot state the required transform\n", n->op_type, n->name);
            continue;
        }
        uint64_t elems = onnx_elem_count(in_dims, in_ndim);
        printf("[derived] layout %s('%s'): activation is ONNX NCHW ", n->op_type, n->name);
        print_shape(in_dims, in_ndim);
        printf(" — canonical CONV/POOL require HWC; transpose (N,C,H,W)->(N,H,W,C) needed on this input (%llu elements, %llu bytes moved)\n", (unsigned long long)elems, (unsigned long long)elems * 4);
        if (is_conv && n->n_inputs >= 2) {
            int64_t w_dims[ONNX_MAX_DIMS]; uint32_t w_ndim;
            if (onnx_find_value_shape(&g, n->inputs[1], w_dims, &w_ndim) == 0 && w_ndim == 4) {
                printf("[derived] layout %s('%s'): weight is ONNX OIHW ", n->op_type, n->name);
                print_shape(w_dims, w_ndim);
                printf(" — matches canonical CONV's own weight layout [COUT, CIN*KH*KW] exactly; no weight transform needed\n");
            }
        }
    }
    if (any_conv_or_pool) {
        printf("[derived] layout: a Reshape/flatten downstream of Conv/MaxPool in this graph was very likely authored against the file's ORIGINAL NCHW activation order — the MatMul weight it feeds needs its rows reordered to match canonical's HWC-flattened order too, not just the activation transpose above (this is a real, previously-hit bug in this project's own hand-authored CNN killer-benchmark, not a hypothetical)\n");
    }

    /* [derived] peak memory: liveness-style — a tensor (initializer, graph
     * input, or node output) is live from its production point to the
     * last node index that consumes it (or the end, if it's a graph
     * output); peak is the max over all points of currently-live bytes.
     * This is a real static analysis, not a guess: same idea as this
     * repo's memory-liveness planner spikes, just over an ONNX graph
     * instead of an ExecStep schedule. */
    {
        uint64_t peak = 0, running = 0;
        /* graph inputs + initializers are live from step -1 */
        for (uint32_t i = 0; i < g.n_inputs; i++) running += onnx_elem_count(g.inputs[i].dims, g.inputs[i].ndim) * 4;
        for (uint32_t i = 0; i < g.n_initializers; i++) running += g.initializers[i].count * 4;
        if (running > peak) peak = running;
        for (uint32_t i = 0; i < g.n_nodes; i++) {
            OnnxNode *n = &g.nodes[i];
            if (n->shape_ok) running += onnx_elem_count(n->out_dims, n->out_ndim) * 4;
            if (running > peak) peak = running;
            /* free any tensor whose last consumer was this node (naive: only node outputs, not initializers/inputs, which we conservatively keep live for the whole graph — this is a floor estimate, not a full liveness planner) */
            for (uint32_t j = 0; j < n->n_inputs; j++) {
                uint32_t last_use = i;
                for (uint32_t k = i + 1; k < g.n_nodes; k++) for (uint32_t m = 0; m < g.nodes[k].n_inputs; m++) if (!strcmp(g.nodes[k].inputs[m], n->inputs[j])) last_use = k;
                if (last_use == i) {
                    for (uint32_t k = 0; k < g.n_nodes; k++) if (k != i)
                        for (uint32_t m = 0; m < g.nodes[k].n_outputs; m++) if (!strcmp(g.nodes[k].outputs[m], n->inputs[j]) && g.nodes[k].shape_ok)
                            running -= onnx_elem_count(g.nodes[k].out_dims, g.nodes[k].out_ndim) * 4;
                }
            }
        }
        printf("[derived] peak memory (weights + live activations, no execution-order optimization applied): %llu bytes\n", (unsigned long long)peak);
    }

    /* [estimated] MACs, per node, from shapes alone — same shape-driven
     * formula family as tensor_merged_mnist_resource_gate.c's
     * estimate_macs(): MatMul/Gemm cost rows*inner*cols, Conv cost
     * out_elements*(CIN*KH*KW) (standard conv MAC count — each output
     * element is one CIN*KH*KW dot product), elementwise ops cost one op
     * per output element. Reshape is excluded from elementwise: it's a
     * pure layout/view operation with no arithmetic, not an op, and
     * counting it would overstate cost (this was a real bug — Conv nodes
     * used to fall into "elementwise" by default, undercounting the
     * dominant cost of any CNN and never surfacing Conv's own MACs at
     * all, caught by running this on the first real Conv-bearing file). */
    double total_macs = 0, total_elementwise = 0;
    for (uint32_t i = 0; i < g.n_nodes; i++) {
        OnnxNode *n = &g.nodes[i];
        if (!n->shape_ok) continue;
        double macs = 0;
        if (!strcmp(n->op_type, "MatMul") || !strcmp(n->op_type, "Gemm")) {
            int64_t da[ONNX_MAX_DIMS], db[ONNX_MAX_DIMS]; uint32_t an, bn;
            if (onnx_find_value_shape(&g, n->inputs[0], da, &an) == 0 && onnx_find_value_shape(&g, n->inputs[1], db, &bn) == 0 && an == 2 && bn == 2) {
                int64_t inner = !strcmp(n->op_type, "Gemm") && n->transA ? da[0] : da[1];
                macs = (double)n->out_dims[0] * (double)inner * (double)n->out_dims[1];
            }
        } else if (!strcmp(n->op_type, "Conv")) {
            int64_t w_dims[ONNX_MAX_DIMS]; uint32_t w_ndim;
            if (n->n_inputs >= 2 && onnx_find_value_shape(&g, n->inputs[1], w_dims, &w_ndim) == 0 && w_ndim == 4) {
                double per_output_dot = (double)w_dims[1] * (double)w_dims[2] * (double)w_dims[3]; /* CIN*KH*KW */
                macs = (double)onnx_elem_count(n->out_dims, n->out_ndim) * per_output_dot;
            }
        } else if (strcmp(n->op_type, "Reshape")) total_elementwise += (double)onnx_elem_count(n->out_dims, n->out_ndim);
        if (macs) printf("[estimated] MACs for %s('%s'): %.0f\n", n->op_type, n->name, macs);
        total_macs += macs;
    }
    printf("[estimated] total MACs (one forward pass, batch=1): %.0f\n", total_macs);
    printf("[estimated] elementwise ops (Add/Relu/MaxPool): %.0f\n", total_elementwise);

    printf("[measured] none (inspect performs no execution — see 'tensorctl verify' for measured cross-engine numbers)\n");
    printf("[unknown] lifetime crossover, warm latency, peak RSS, deploy footprint (require actually running this model against a competing engine)\n");

    if (g.n_unsupported) {
        for (uint32_t i = 0; i < g.n_unsupported; i++) printf("[unsupported] %s\n", g.unsupported[i].message);
    } else {
        printf("[unsupported] none — every node in this graph is in the v0 subset (MatMul/Gemm/Add/Relu/Conv/MaxPool/Reshape)\n");
    }

    unsigned build_unsupported = 0;
    for (uint32_t i = 0; i < g.n_nodes; i++) {
        if (!(onnx_op_capability(g.nodes[i].op_type) & ONNX_CAP_BUILD)) {
            printf("[build-unsupported] node '%s' (%s): importable for inspection, but native lowering is not implemented\n",
                   g.nodes[i].name, g.nodes[i].op_type);
            build_unsupported++;
        }
    }
    if (!build_unsupported) printf("[build-unsupported] none — every imported node has native lowering\n");

    return g.n_unsupported ? 1 : 0;
}
