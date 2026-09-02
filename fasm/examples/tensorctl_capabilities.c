/* Machine-readable ONNX capability surface. Keep product wiring thin: the
 * authoritative table lives beside the shared importer. */
#include <stdio.h>
#include "tensor_onnx_import.h"

int run_capabilities(int argc, char **argv) {
    (void)argv;
    if (argc != 1) {
        fprintf(stderr, "usage: tensorctl capabilities\n");
        return 2;
    }
    puts("# tensorctl-onnx-capabilities/v1");
    puts("op_type\timportable\tbuildable");
    unsigned n = (unsigned)(sizeof onnx_op_capabilities / sizeof onnx_op_capabilities[0]);
    for (unsigned i = 0; i < n; i++)
        printf("%s\t%s\t%s\n", onnx_op_capabilities[i].op_type,
               onnx_op_capabilities[i].flags & ONNX_CAP_IMPORT ? "yes" : "no",
               onnx_op_capabilities[i].flags & ONNX_CAP_BUILD ? "yes" : "no");
    return 0;
}
