#include <stdio.h>
#include <string.h>

/* tensorctl: v0.1 usable entry point into the graph-planned tensor training
 * engine. tensor_mlp_executor_spike.h and tensor_transformer_executor_spike.h
 * both define Block/ExecStep/Context/etc with the same names, so they can't
 * share a translation unit — each model gets its own .c file, only this
 * dispatcher sees both entry points. */
extern int run_mlp(int argc, char **argv);
extern int run_transformer(int argc, char **argv);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: tensorctl <mlp|transformer> [--epochs N] [--memory-budget N]\n"
            "  mlp:         trains the 2-4-1 XOR MLP through the shared executor\n"
            "  transformer: trains the T=3,M=4,H=2,D=2,F=6 encoder block through the shared executor\n"
            "               --memory-budget lets you watch the save-vs-rematerialize decision flip\n");
        return 2;
    }
    if (!strcmp(argv[1], "mlp")) return run_mlp(argc - 1, argv + 1);
    if (!strcmp(argv[1], "transformer")) return run_transformer(argc - 1, argv + 1);
    fprintf(stderr, "tensorctl: unknown model '%s' (expected mlp or transformer)\n", argv[1]);
    return 2;
}
