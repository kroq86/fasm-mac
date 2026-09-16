/* Portable-C arm64 alternative to tensor_transformer_executor_f32.asm's
 * tensor_transformer_steps_execute. That file is genuine x86-64 assembly
 * (FASM does not target arm64), but the function itself is a generic
 * function-pointer dispatch loop with no architecture-specific instructions:
 * call run(context) for each step in order, stop and return the first
 * non-zero result, otherwise return 0. This mirrors the identical
 * portable-C definition already used by the arm64-native killer-workload
 * spikes (e.g. tensor_killer_mnist_native.c), which take this same
 * approach instead of linking the .asm object, for the same reason.
 *
 * Not a replacement for tensor_transformer_executor_f32.asm in general:
 * every other check_*_spike.sh gate keeps assembling and linking that
 * file unchanged. This file exists only for arm64 builds that choose not
 * to link the x86-64 object (see scripts/build-kv-handoff-arm64.sh).
 */
#include "tensor_semantic_compiler.h"

int tensor_transformer_steps_execute(const ExecStep *s, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        int rc = s[i].run(s[i].context);
        if (rc) return rc;
    }
    return 0;
}
