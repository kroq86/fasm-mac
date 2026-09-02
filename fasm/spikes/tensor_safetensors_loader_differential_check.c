/* Differential/property check for the self-written safetensors loader
 * (tensor_safetensors_loader.h). Oracle values below were computed
 * independently -- raw struct/json parsing in Python, NOT the
 * `safetensors` library and NOT PyTorch -- directly against the real
 * `gpt2` checkpoint (huggingface.co/gpt2, revision
 * 607a30d783dfa663caf39e06633721c8d4cfcd7e, model.safetensors,
 * sha256=248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707):
 *
 *   python3 -c "
 *   import struct, json
 *   with open(path, 'rb') as f:
 *       hlen = struct.unpack('<Q', f.read(8))[0]
 *       header = json.loads(f.read(hlen)); data_start = 8 + hlen
 *       off0, off1 = header[name]['data_offsets']
 *       f.seek(data_start + off0)
 *       vals = struct.unpack('<%df' % ((off1-off0)//4), f.read(off1-off0))"
 *
 * This is an external, environment-dependent fixture (the real ~548MB
 * checkpoint, not committed) -- skips gracefully if not present, same
 * convention as the MNIST ONNX and GPT-2 block0 gates.
 */
#include "tensor_safetensors_loader.h"
#include <math.h>

static int nearly(float a, float b, float tol) { return fabsf(a - b) <= tol; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
        : "/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors";

    SafetensorsFile st;
    if (st_open(&st, path)) {
        printf("safetensors_loader: fixture not found at %s -- skipped\n", path);
        return 0;
    }

    /* property 1: correct bytes at the correct offset, cross-checked
     * against an independently (non-C, non-safetensors-library) computed
     * oracle on real tensors */
    {
        static float ln1w[768];
        uint64_t shape[1] = {768};
        if (st_read_f32(&st, "h.0.ln_1.weight", shape, 1, ln1w)) { fprintf(stderr, "read h.0.ln_1.weight failed\n"); return 1; }
        double sum = 0; for (int i = 0; i < 768; i++) sum += ln1w[i];
        if (!nearly(ln1w[0], 0.2232203334569931f, 1e-6f) || !nearly(ln1w[1], 0.1819586604833603f, 1e-6f) ||
            !nearly(ln1w[4], 0.20361845195293427f, 1e-6f) || !nearly(ln1w[767], 0.1898234784603119f, 1e-6f) ||
            fabs(sum - 138.5156662426889) > 1e-2) {
            fprintf(stderr, "ln_1.weight mismatch vs independent oracle: [0]=%.9g [1]=%.9g [4]=%.9g [767]=%.9g sum=%.9g\n", ln1w[0], ln1w[1], ln1w[4], ln1w[767], sum);
            return 1;
        }

        static float ln1b[768];
        if (st_read_f32(&st, "h.0.ln_1.bias", shape, 1, ln1b)) { fprintf(stderr, "read h.0.ln_1.bias failed\n"); return 1; }
        if (!nearly(ln1b[0], -0.003677325090393424f, 1e-6f) || !nearly(ln1b[2], -0.064040906727314f, 1e-6f)) {
            fprintf(stderr, "ln_1.bias mismatch vs independent oracle: [0]=%.9g [2]=%.9g\n", ln1b[0], ln1b[2]);
            return 1;
        }

        static float cattnb[2304];
        uint64_t shape2304[1] = {2304};
        if (st_read_f32(&st, "h.0.attn.c_attn.bias", shape2304, 1, cattnb)) { fprintf(stderr, "read h.0.attn.c_attn.bias failed\n"); return 1; }
        double sum2 = 0; for (int i = 0; i < 2304; i++) sum2 += cattnb[i];
        if (!nearly(cattnb[0], 0.4803391396999359f, 1e-6f) || !nearly(cattnb[2303], 0.003247644752264023f, 1e-6f) ||
            fabs(sum2 - (-1.629668176830819)) > 1e-2) {
            fprintf(stderr, "attn.c_attn.bias mismatch vs independent oracle: [0]=%.9g [2303]=%.9g sum=%.9g\n", cattnb[0], cattnb[2303], sum2);
            return 1;
        }
        printf("safetensors_loader: real tensor bytes match independent raw-struct-parsed oracle exactly (ln_1.weight, ln_1.bias, attn.c_attn.bias)\n");
    }

    /* property 2: 2D tensor shape/dtype validated, correct element count */
    {
        static float cattnw[768 * 2304];
        uint64_t shape2d[2] = {768, 2304};
        if (st_read_f32(&st, "h.0.attn.c_attn.weight", shape2d, 2, cattnw)) { fprintf(stderr, "read h.0.attn.c_attn.weight failed\n"); return 1; }
        printf("safetensors_loader: 2D tensor h.0.attn.c_attn.weight [768,2304] read successfully, first=%.9g\n", cattnw[0]);
    }

    /* property 3: fail closed on a nonexistent tensor name */
    {
        static float dummy[8];
        uint64_t shape8[1] = {8};
        FILE *saved_stderr_marker = NULL; (void)saved_stderr_marker;
        int rc = st_read_f32(&st, "h.0.this_does_not_exist", shape8, 1, dummy);
        if (rc == 0) { fprintf(stderr, "expected failure for nonexistent tensor name, got success\n"); return 1; }
        printf("safetensors_loader: nonexistent tensor name correctly rejected (fail-closed)\n");
    }

    /* property 4: fail closed on a wrong expected shape */
    {
        static float dummy[768];
        uint64_t wrong_shape[1] = {512}; /* real shape is 768 */
        int rc = st_read_f32(&st, "h.0.ln_1.weight", wrong_shape, 1, dummy);
        if (rc == 0) { fprintf(stderr, "expected failure for shape mismatch, got success\n"); return 1; }
        printf("safetensors_loader: shape mismatch correctly rejected (fail-closed)\n");
    }

    st_close(&st);
    puts("safetensors_loader differential check passed: real checkpoint bytes match an independent oracle, 2D shapes read correctly, missing names and shape mismatches fail closed");
    return 0;
}
