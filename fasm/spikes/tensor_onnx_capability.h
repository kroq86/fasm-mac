#ifndef TENSOR_ONNX_CAPABILITY_H
#define TENSOR_ONNX_CAPABILITY_H

/* Shared contract for the external ONNX vertical slice. IMPORT means the
 * graph can be parsed and explained; BUILD means native lowering exists. */
enum { ONNX_CAP_IMPORT = 1u << 0, ONNX_CAP_BUILD = 1u << 1 };
typedef struct { const char *op_type; unsigned flags; } OnnxOpCapability;
static const OnnxOpCapability onnx_op_capabilities[] = {
    {"MatMul", ONNX_CAP_IMPORT | ONNX_CAP_BUILD},
    {"Gemm", ONNX_CAP_IMPORT | ONNX_CAP_BUILD},
    {"Add", ONNX_CAP_IMPORT | ONNX_CAP_BUILD},
    {"Relu", ONNX_CAP_IMPORT | ONNX_CAP_BUILD},
    {"Conv", ONNX_CAP_IMPORT | ONNX_CAP_BUILD},
    {"MaxPool", ONNX_CAP_IMPORT | ONNX_CAP_BUILD},
    {"Reshape", ONNX_CAP_IMPORT | ONNX_CAP_BUILD},
};
static unsigned onnx_op_capability(const char *op_type) {
    unsigned n = (unsigned)(sizeof onnx_op_capabilities / sizeof onnx_op_capabilities[0]);
    for (unsigned i = 0; i < n; i++)
        if (!strcmp(onnx_op_capabilities[i].op_type, op_type)) return onnx_op_capabilities[i].flags;
    return 0;
}
#endif
