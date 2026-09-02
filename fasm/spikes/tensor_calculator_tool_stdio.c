/* Standalone "external tool" binary for the embedded-vs-tool-calling
 * comparison. Reads a fixed 68-byte frame from stdin: 44 bytes
 * ProposalWireV1 + 24 bytes OperandRequest (operand[2] int64 + uint64
 * context_fingerprint, naturally 8-aligned, no padding). Calls the exact
 * same replay_wire() the embedded path calls -- same executor, same
 * fail-closed semantics, same wire ABI. Only the delivery mechanism
 * differs: this binary is meant to be fork+exec'd per call with the frame
 * piped to its stdin, mirroring how a real LLM tool-call subprocess
 * (e.g. a Python code-interpreter sandbox) is invoked.
 *
 * Protocol: on success, writes exactly 8 bytes (native-endian int64
 * answer) to stdout. With --persistent it then reads the next frame and
 * exits cleanly only at an exact frame boundary EOF. On any rejection, writes NOTHING to
 * stdout and exits with a nonzero code derived from replay_wire's
 * negative return (|rc|, clamped to 1..255) -- fail-closed: the caller
 * must check the exit code before trusting any stdout bytes, and even
 * then must check it received exactly 8 bytes.
 */
#define NS_EXECUTOR_NO_SELFTEST
#define main calculator_tool_unused_main
#include "tensor_neurosymbolic_calculator_check.c"
#undef main

#include <stdlib.h>
#include <unistd.h>

enum { FRAME_SIZE = sizeof(ProposalWireV1) + sizeof(OperandRequest) };

static ssize_t read_all(int fd, void *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, (char *)buf + got, n - got);
        if (r <= 0) return r < 0 ? -1 : (ssize_t)got;
        got += (size_t)r;
    }
    return (ssize_t)got;
}
static ssize_t write_all(int fd, const void *buf, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = write(fd, (const char *)buf + sent, n - sent);
        if (w <= 0) return -1;
        sent += (size_t)w;
    }
    return (ssize_t)sent;
}

int main(int argc, char **argv) {
    /* Real tool-calling protocols must tell the tool which model/policy it
     * is trusting -- this is not a shortcut, it's the external-process
     * analogue of the embedded path sharing MODEL_FP in-process for free.
     * argv[1] is the trained model's fingerprint as a hex string. */
    if (argc >= 2) MODEL_FP = strtoull(argv[1], NULL, 16);

    int persistent = argc >= 3 && !strcmp(argv[2], "--persistent");
    for (;;) {
        unsigned char frame[FRAME_SIZE];
        ssize_t got = read_all(0, frame, sizeof frame);
        if (persistent && got == 0) return 0;
        if (got != (ssize_t)sizeof frame) return 60; /* short frame: reject, write nothing */

        ProposalWireV1 wire;
        OperandRequest req;
        memcpy(&wire, frame, sizeof wire);
        memcpy(&req, frame + sizeof wire, sizeof req);

        int64_t answer;
        uint16_t trace[MAX_TRACE], trace_n = 0;
        int rc = replay_wire(&wire, sizeof wire, &req, &answer, trace, &trace_n);
        if (rc) { int code = -rc; if (code < 1) code = 1; if (code > 255) code = 255; return code; }

        if (write_all(1, &answer, sizeof answer) != (ssize_t)sizeof answer) return 61;
        if (!persistent) return 0;
    }
}
