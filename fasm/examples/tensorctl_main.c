#include <stdio.h>
#include <string.h>

/* tensorctl: v0.1 usable entry point into the graph-planned tensor training
 * engine. tensor_mlp_executor_spike.h and tensor_transformer_executor_spike.h
 * both define Block/ExecStep/Context/etc with the same names, so they can't
 * share a translation unit — each model gets its own .c file, only this
 * dispatcher sees both entry points. */
extern int run_mlp(int argc, char **argv);
extern int run_transformer(int argc, char **argv);
extern int run_plan_diff(int argc, char **argv);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: tensorctl <mlp|transformer> [--plan] ...\n"
            "       tensorctl plan transformer --export FILE.tsv [planning options]\n"
            "       tensorctl plan diff PLAN_A.tsv PLAN_B.tsv [--machine]\n"
            "  mlp:          trains the 2-4-1 XOR MLP through the shared executor\n"
            "  transformer:  trains the T=3,M=4,H=2,D=2,F=6 encoder block through the shared executor\n"
            "  --plan:       print the execution/memory/layout plan and exit, no training run\n"
            "  --memory-budget lets you watch the save-vs-rematerialize decision flip (transformer only)\n"
            "  --explain-decisions/--show-alternatives: render the planner decision trace\n"
            "  --counterfactual-budget=N: show the memory choice under another budget\n"
            "  --reprofile:  force a fresh layout-decision benchmark, ignoring any cached one (transformer only)\n"
            "  --no-profile-cache: never read or write the layout-decision profile cache (transformer only)\n");
        return 2;
    }
    if (!strcmp(argv[1], "mlp")) return run_mlp(argc - 1, argv + 1);
    if (!strcmp(argv[1], "transformer")) return run_transformer(argc - 1, argv + 1);
    if (!strcmp(argv[1], "plan") && argc >= 3 && !strcmp(argv[2], "diff")) return run_plan_diff(argc - 3, argv + 3);
    if (!strcmp(argv[1], "plan") && argc >= 3 && !strcmp(argv[2], "transformer")) {
        char *forwarded[64];
        if (argc + 1 > (int)(sizeof forwarded / sizeof forwarded[0])) return 2;
        forwarded[0] = argv[2]; forwarded[1] = "--plan";
        for (int i = 3; i < argc; i++) forwarded[i - 1] = argv[i];
        return run_transformer(argc - 1, forwarded);
    }
    fprintf(stderr, "tensorctl: unknown model '%s' (expected mlp or transformer)\n", argv[1]);
    return 2;
}
