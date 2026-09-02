#include "tensor_semantic_compiler.h"
#include "tensor_gpt2_forward.h"
#include "tensor_gpt2_bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#if M != 768 || H != 12 || D != 64 || QW != 2304 || MAXCACHE != 64
#error "tensorctl_gpt2.c requires the real GPT-2 124M shape macros"
#endif

enum { GPT2_CLI_MAX_TOKENS = 64 };
enum { CLI_X, CLI_CACHE, CLI_ATT, CLI_NODES };

typedef struct {
    Node graph[CLI_NODES];
    ExecStep steps[8];
    Context contexts[8];
    uint32_t step_count;
    float qkv[QW], qkv_grad[QW];
    float keys[MAXCACHE * M], keys_grad[MAXCACHE * M], values[MAXCACHE * M];
    float output[M], output_grad[M];
} Gpt2CliLayer;

static Gpt2CliLayer cli_layers[GPT2_NLAYER];
static Gpt2Weights cli_weights;
static Gpt2Bpe cli_bpe;
typedef void (*Gpt2Matmul)(const float *, int, int, const float *, const float *, int, float *);
static Gpt2Matmul cli_matmul = gpt2_matmul_bias;
#ifdef __APPLE__
void gpt2_matmul_bias_accelerate(const float *, int, int, const float *, const float *, int, float *);
#endif

static int cli_layer_setup(Gpt2CliLayer *layer) {
    memset(layer, 0, sizeof *layer);
    layer->graph[CLI_X] = (Node){LEAF, NONE, NONE, INPUT,
        {layer->qkv, layer->qkv_grad, NULL, 1, QW, 0}};
    layer->graph[CLI_CACHE] = (Node){LEAF, NONE, NONE, INPUT,
        {layer->keys, layer->keys_grad, layer->values, MAXCACHE, M, 0}};
    layer->graph[CLI_ATT] = (Node){CAUSAL_ATTENTION_CACHED, CLI_X, CLI_CACHE, TEMP,
        {layer->output, layer->output_grad, NULL, 1, M, 0}};
    return compile(layer->graph, CLI_NODES, 0.0f, layer->steps, layer->contexts,
                   (uint32_t)(sizeof layer->steps / sizeof layer->steps[0]), &layer->step_count);
}

static void cli_prefill(const int *tokens, int count) {
    if (count <= 0) return;
    static float hidden[GPT2_MAXT * GPT2_M];
    static float ln1[GPT2_MAXT * GPT2_M], qkv[GPT2_MAXT * GPT2_QW], attn[GPT2_MAXT * GPT2_M];
    static float proj[GPT2_MAXT * GPT2_M], residual[GPT2_MAXT * GPT2_M], ln2[GPT2_MAXT * GPT2_M];
    static float fc[GPT2_MAXT * GPT2_F], activated[GPT2_MAXT * GPT2_F], fcproj[GPT2_MAXT * GPT2_M];
    for (int t = 0; t < count; t++) for (int m = 0; m < GPT2_M; m++)
        hidden[t * GPT2_M + m] = cli_weights.wte[tokens[t] * GPT2_M + m] + cli_weights.wpe[t * GPT2_M + m];
    for (int block = 0; block < GPT2_NLAYER; block++) {
        const Gpt2BlockWeights *bw = &cli_weights.blk[block];
        gpt2_layernorm_raw(hidden, count, GPT2_M, ln1);
        cli_matmul(ln1, count, GPT2_M, bw->attn_w, bw->attn_b, GPT2_QW, qkv);
        gpt2_causal_attention(qkv, count, attn);
        for (int t = 0; t < count; t++) {
            memcpy(&cli_layers[block].keys[t * M], &qkv[t * GPT2_QW + GPT2_M], sizeof(float) * M);
            memcpy(&cli_layers[block].values[t * M], &qkv[t * GPT2_QW + 2 * GPT2_M], sizeof(float) * M);
        }
        cli_layers[block].graph[CLI_CACHE].tensor.aux_count = (uint32_t)count;
        cli_matmul(attn, count, GPT2_M, bw->projw, bw->projb, GPT2_M, proj);
        for (int i = 0; i < count * GPT2_M; i++) residual[i] = hidden[i] + proj[i];
        gpt2_layernorm_raw(residual, count, GPT2_M, ln2);
        cli_matmul(ln2, count, GPT2_M, bw->fc_w, bw->fc_b, GPT2_F, fc);
        gpt2_gelu(fc, count * GPT2_F, activated);
        cli_matmul(activated, count, GPT2_F, bw->fcproj_w, bw->fcproj_b, GPT2_M, fcproj);
        for (int i = 0; i < count * GPT2_M; i++) hidden[i] = residual[i] + fcproj[i];
    }
}

static int cli_decode(int token, int position, float *logits) {
    float hidden[M];
    for (int m = 0; m < M; m++) hidden[m] = cli_weights.wte[token * M + m] + cli_weights.wpe[position * M + m];
    for (int block = 0; block < GPT2_NLAYER; block++) {
        const Gpt2BlockWeights *bw = &cli_weights.blk[block];
        Gpt2CliLayer *layer = &cli_layers[block];
        float ln1[M]; gpt2_layernorm_raw(hidden, 1, M, ln1);
        cli_matmul(ln1, 1, M, bw->attn_w, bw->attn_b, QW, layer->qkv);
        if (tensor_transformer_steps_execute(layer->steps, layer->step_count)) return -1;
        float proj[M]; cli_matmul(layer->output, 1, M, bw->projw, bw->projb, M, proj);
        float residual[M]; for (int i = 0; i < M; i++) residual[i] = hidden[i] + proj[i];
        float ln2[M]; gpt2_layernorm_raw(residual, 1, M, ln2);
        float fc[F]; cli_matmul(ln2, 1, M, bw->fc_w, bw->fc_b, F, fc);
        float activated[F]; gpt2_gelu(fc, F, activated);
        float fcproj[M]; cli_matmul(activated, 1, F, bw->fcproj_w, bw->fcproj_b, M, fcproj);
        for (int i = 0; i < M; i++) hidden[i] = residual[i] + fcproj[i];
    }
    float final_hidden[M];
    gpt2_layernorm_raw(hidden, 1, M, final_hidden);
    cli_matmul(final_hidden, 1, M, cli_weights.lnf_w_folded,
               cli_weights.lnf_b_folded, GPT2_VOCAB, logits);
    return 0;
}

static int argmax(const float *values, int count) {
    int best = 0;
    for (int i = 1; i < count; i++) if (values[i] > values[best]) best = i;
    return best;
}

static int ascii_only(const char *text) {
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) if (*p >= 128) return 0;
    return 1;
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static double peak_rss_mb(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage)) return -1.0;
    return (double)usage.ru_maxrss / (1024.0 * 1024.0);
}

int run_gpt2(int argc, char **argv) {
    const char *model = getenv("GPT2_SAFETENSORS");
    const char *tokenizer = getenv("GPT2_TOKENIZER_FIXTURE_DIR");
    const char *prompt = NULL;
    const char *backend = "accelerate";
    int generated = 12;
    if (!model) model = "/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors";
    if (!tokenizer) tokenizer = "scratchpad/gpt2_tokenizer_fixture";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) generated = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) tokenizer = argv[++i];
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
        else {
            fprintf(stderr, "usage: tensorctl gpt2 --prompt TEXT [--tokens N] [--backend accelerate|scalar] [--model FILE] [--tokenizer DIR]\n");
            return 2;
        }
    }
    if (!strcmp(backend, "scalar")) cli_matmul = gpt2_matmul_bias;
#ifdef __APPLE__
    else if (!strcmp(backend, "accelerate")) cli_matmul = gpt2_matmul_bias_accelerate;
#endif
    else { fprintf(stderr, "tensorctl gpt2: unsupported backend: %s\n", backend); return 2; }
    if (!prompt || !*prompt) { fprintf(stderr, "tensorctl gpt2: --prompt must not be empty\n"); return 2; }
    if (!ascii_only(prompt)) { fprintf(stderr, "tensorctl gpt2: only ASCII prompts are currently supported\n"); return 2; }
    if (generated < 1 || generated >= GPT2_CLI_MAX_TOKENS) { fprintf(stderr, "tensorctl gpt2: --tokens must be in [1,63]\n"); return 2; }

    char vocab[1024], merges[1024];
    if (snprintf(vocab, sizeof vocab, "%s/vocab.json", tokenizer) >= (int)sizeof vocab ||
        snprintf(merges, sizeof merges, "%s/merges.txt", tokenizer) >= (int)sizeof merges) {
        fprintf(stderr, "tensorctl gpt2: tokenizer path too long\n"); return 2;
    }
    const double load_start = monotonic_ms();
    fprintf(stderr, "Loading GPT-2 124M...\n");
    if (gpt2_load_weights(&cli_weights, model, "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707")) {
        fprintf(stderr, "tensorctl gpt2: model load or SHA-256 verification failed\n"); return 3;
    }
    const double model_loaded = monotonic_ms();
    if (gpt2_bpe_load(&cli_bpe, vocab, merges)) {
        fprintf(stderr, "tensorctl gpt2: tokenizer load failed under %s\n", tokenizer); return 3;
    }
    const double tokenizer_loaded = monotonic_ms();
    int ids[GPT2_CLI_MAX_TOKENS];
    int prompt_count = gpt2_bpe_encode(&cli_bpe, prompt, (int)strlen(prompt), ids, GPT2_CLI_MAX_TOKENS);
    if (prompt_count < 1) { fprintf(stderr, "tensorctl gpt2: prompt tokenization failed\n"); return 3; }
    if (prompt_count + generated > GPT2_CLI_MAX_TOKENS) {
        fprintf(stderr, "tensorctl gpt2: prompt (%d tokens) + generation (%d) exceeds context capacity %d\n",
                prompt_count, generated, GPT2_CLI_MAX_TOKENS); return 2;
    }
    for (int i = 0; i < GPT2_NLAYER; i++) if (cli_layer_setup(&cli_layers[i])) {
        fprintf(stderr, "tensorctl gpt2: graph compilation failed\n"); return 3;
    }
    const double inference_start = monotonic_ms();
    cli_prefill(ids, prompt_count - 1);
    const double prefill_done = monotonic_ms();
    static float logits[GPT2_VOCAB];
    double first_token_done = 0.0;
    for (int step = 0; step < generated; step++) {
        int position = prompt_count + step - 1;
        if (cli_decode(ids[position], position, logits)) { fprintf(stderr, "tensorctl gpt2: execution failed\n"); return 3; }
        ids[prompt_count + step] = argmax(logits, GPT2_VOCAB);
        if (step == 0) first_token_done = monotonic_ms();
    }
    const double inference_done = monotonic_ms();
    char output[16384];
    int bytes = gpt2_bpe_decode(&cli_bpe, ids, prompt_count + generated, output, (int)sizeof output - 1);
    if (bytes < 0) { fprintf(stderr, "tensorctl gpt2: output decoding failed\n"); return 3; }
    output[bytes] = '\0';
    printf("%s\n", output);
    const double decode_ms = inference_done - first_token_done;
    const double decode_tps = generated > 1 && decode_ms > 0.0
        ? (double)(generated - 1) * 1000.0 / decode_ms : 0.0;
    fprintf(stderr,
        "\nGPT-2 runtime metadata\n"
        "  backend: %s\n"
        "  model: gpt2-124m (layers=%d hidden=%d heads=%d vocab=%d)\n"
        "  model_sha256: 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707\n"
        "  prompt_tokens: %d\n"
        "  generated_tokens: %d\n"
        "  context_tokens: %d/%d\n"
        "  model_load_ms: %.3f\n"
        "  tokenizer_load_ms: %.3f\n"
        "  prefill_ms: %.3f\n"
        "  ttft_ms: %.3f\n"
        "  decode_ms_after_first: %.3f\n"
        "  decode_tokens_per_sec: %.3f\n"
        "  inference_total_ms: %.3f\n"
        "  peak_rss_mb: %.3f\n"
        "  sampling: greedy_argmax\n"
        "  kv_cache: canonical_persistent (capacity=%d)\n",
        backend, GPT2_NLAYER, GPT2_M, GPT2_H, GPT2_VOCAB,
        prompt_count, generated, prompt_count + generated, GPT2_CLI_MAX_TOKENS,
        model_loaded - load_start, tokenizer_loaded - model_loaded,
        prefill_done - inference_start, first_token_done - inference_start,
        decode_ms, decode_tps, inference_done - inference_start,
        peak_rss_mb(), MAXCACHE);
    return 0;
}
