#ifndef TENSOR_GPT2_BPE_H
#define TENSOR_GPT2_BPE_H
/* Self-written GPT-2 byte-level BPE tokenizer. No external library. Loads
 * the real, pinned `vocab.json`/`merges.txt` (fetched by
 * scripts/fetch-gpt2-tokenizer.sh, same repo/revision as the real GPT-2
 * 124M checkpoint) and implements the actual published algorithm:
 *   1. a fixed byte<->unicode-codepoint table (printable ASCII/Latin-1
 *      supplement bytes map to themselves; every other byte maps to a
 *      codepoint in 256..323) -- computed here by the same procedure as
 *      the reference, not hardcoded by hand;
 *   2. pre-tokenization: split input text into chunks matching (in this
 *      priority order) the 7 alternatives of GPT-2's own regex
 *          's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 *      SCOPE DECISION (explicit, made with the user): \p{L}/\p{N} are
 *      restricted here to ASCII letters/digits, not full Unicode
 *      categories (no Unicode category tables in this from-scratch
 *      project). Byte-encoder table itself is still exact/complete --
 *      only the pre-tokenization CLASSIFICATION of the input text is
 *      ASCII-scoped. Round-trip correctness is therefore only claimed
 *      for ASCII input; the byte-fallback path still works for any byte
 *      value once bytes reach BPE (a non-ASCII byte just always falls
 *      into the "other" class rather than being letter/number-classified).
 *   3. byte-level BPE merge, using the real, rank-ordered merges.txt;
 *   4. vocab lookup of the final merged pieces for token ids (encode) and
 *      the inverse (decode).
 *
 * Data structures: an open-addressing hash table (FNV-1a, linear probe)
 * for vocab string->id and merge-pair->rank lookups; all piece bytes for
 * both vocab and merges live concatenated in two flat buffers, indexed by
 * (offset,len) spans -- no per-string allocation.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    BPE_MAX_VOCAB = 51200,
    BPE_MAX_MERGES = 51200,
    BPE_HASH_SLOTS = 131072, /* power of 2, > 2x max(vocab,merges) */
    BPE_VOCAB_BUF_BYTES = 2 * 1024 * 1024,
    BPE_MERGES_BUF_BYTES = 2 * 1024 * 1024,
    BPE_MAX_TOKEN_PIECES = 256, /* per single pre-tokenized chunk */
};

typedef struct { uint32_t off, len; } BpeSpan;

typedef struct {
    char vocab_buf[BPE_VOCAB_BUF_BYTES];
    uint32_t vocab_buf_used;
    BpeSpan vocab_span[BPE_MAX_VOCAB]; /* index == token id */
    int vocab_count;
    int32_t vocab_hash[BPE_HASH_SLOTS]; /* -1 empty, else token id */

    char merges_buf[BPE_MERGES_BUF_BYTES];
    uint32_t merges_buf_used;
    BpeSpan merge_a[BPE_MAX_MERGES], merge_b[BPE_MAX_MERGES]; /* rank == index */
    int merge_count;
    int32_t merge_hash[BPE_HASH_SLOTS]; /* -1 empty, else rank */

    char byte_enc[256][4];
    uint8_t byte_enc_len[256];
} Gpt2Bpe;

static uint32_t bpe_fnv1a(const char *s, uint32_t len) {
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}

static void bpe_encode_utf8(uint32_t cp, char *out, uint8_t *out_len) {
    if (cp < 0x80) { out[0] = (char)cp; *out_len = 1; }
    else { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); *out_len = 2; }
}

static void bpe_build_byte_table(Gpt2Bpe *bpe) {
    uint8_t mapped[256] = {0};
    uint32_t cps[256];
    for (int b = 33; b <= 126; b++) { cps[b] = (uint32_t)b; mapped[b] = 1; }
    for (int b = 161; b <= 172; b++) { cps[b] = (uint32_t)b; mapped[b] = 1; }
    for (int b = 174; b <= 255; b++) { cps[b] = (uint32_t)b; mapped[b] = 1; }
    uint32_t n = 0;
    for (int b = 0; b < 256; b++) if (!mapped[b]) { cps[b] = 256 + n; n++; }
    for (int b = 0; b < 256; b++) bpe_encode_utf8(cps[b], bpe->byte_enc[b], &bpe->byte_enc_len[b]);
}

/* JSON string parser with \uXXXX support (needed: vocab.json escapes the
 * byte-encoder's shifted codepoints as Ā-Ń-ish). Appends decoded
 * UTF-8 bytes to out (cap bytes); returns byte count written, or -1 on
 * malformed/overflow. Advances *p past the closing quote. */
static int bpe_parse_json_string(const char **p, char *out, int cap) {
    const char *s = *p;
    if (*s != '"') return -1;
    s++;
    int n = 0;
    while (*s && *s != '"') {
        if (*s == '\\') {
            s++;
            if (*s == 'u') {
                s++;
                uint32_t cp = 0;
                for (int i = 0; i < 4; i++) {
                    char c = s[i];
                    uint32_t d;
                    if (c >= '0' && c <= '9') d = (uint32_t)(c - '0');
                    else if (c >= 'a' && c <= 'f') d = (uint32_t)(c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F') d = (uint32_t)(c - 'A' + 10);
                    else return -1;
                    cp = (cp << 4) | d;
                }
                s += 4;
                char enc[4]; uint8_t elen;
                bpe_encode_utf8(cp, enc, &elen);
                if (n + elen > cap) return -1;
                memcpy(out + n, enc, elen); n += (int)elen;
                continue;
            }
            char c;
            switch (*s) { case '"': c = '"'; break; case '\\': c = '\\'; break; case '/': c = '/'; break;
                default: return -1; }
            if (n + 1 > cap) return -1;
            out[n++] = c; s++;
            continue;
        }
        if (n + 1 > cap) return -1;
        out[n++] = *s; s++;
    }
    if (*s != '"') return -1;
    *p = s + 1;
    return n;
}

static void bpe_hash_insert(int32_t *table, const char *buf, const BpeSpan *spans, int idx, const char *key, uint32_t keylen) {
    (void)buf; (void)spans;
    uint32_t h = bpe_fnv1a(key, keylen) & (BPE_HASH_SLOTS - 1);
    while (table[h] >= 0) h = (h + 1) & (BPE_HASH_SLOTS - 1);
    table[h] = idx;
}

static int bpe_load_vocab(Gpt2Bpe *bpe, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *raw = malloc((size_t)len + 1);
    if (!raw || fread(raw, 1, (size_t)len, f) != (size_t)len) { fclose(f); free(raw); return -1; }
    raw[len] = '\0';
    fclose(f);

    for (int i = 0; i < BPE_HASH_SLOTS; i++) bpe->vocab_hash[i] = -1;
    bpe->vocab_count = 0;
    bpe->vocab_buf_used = 0;

    const char *p = raw;
    while (*p && *p != '{') p++;
    if (*p != '{') { free(raw); return -1; }
    p++;
    for (;;) {
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r' || *p == ',') p++;
        if (*p == '}') break;
        if (*p != '"') { free(raw); return -1; }
        char keybuf[512];
        int klen = bpe_parse_json_string(&p, keybuf, sizeof keybuf);
        if (klen < 0) { free(raw); return -1; }
        while (*p == ' ') p++;
        if (*p != ':') { free(raw); return -1; }
        p++;
        while (*p == ' ') p++;
        char *end;
        long id = strtol(p, &end, 10);
        if (end == p || id < 0 || id >= BPE_MAX_VOCAB) { free(raw); return -1; }
        p = end;

        if (bpe->vocab_buf_used + (uint32_t)klen > BPE_VOCAB_BUF_BYTES) { free(raw); return -1; }
        memcpy(bpe->vocab_buf + bpe->vocab_buf_used, keybuf, (size_t)klen);
        bpe->vocab_span[id] = (BpeSpan){bpe->vocab_buf_used, (uint32_t)klen};
        bpe_hash_insert(bpe->vocab_hash, bpe->vocab_buf, bpe->vocab_span, (int)id, keybuf, (uint32_t)klen);
        bpe->vocab_buf_used += (uint32_t)klen;
        if ((int)id + 1 > bpe->vocab_count) bpe->vocab_count = (int)id + 1;
    }
    free(raw);
    return 0;
}

static int bpe_load_merges(Gpt2Bpe *bpe, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    for (int i = 0; i < BPE_HASH_SLOTS; i++) bpe->merge_hash[i] = -1;
    bpe->merge_count = 0;
    bpe->merges_buf_used = 0;

    char line[512];
    int first = 1;
    while (fgets(line, sizeof line, f)) {
        int n = (int)strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) n--;
        line[n] = '\0';
        if (first) { first = 0; if (line[0] == '#') continue; }
        if (n == 0) continue;
        char *space = strchr(line, ' ');
        if (!space) { fclose(f); return -1; }
        int alen = (int)(space - line);
        int blen = n - alen - 1;
        if (alen <= 0 || blen <= 0) { fclose(f); return -1; }
        int idx = bpe->merge_count;
        if (idx >= BPE_MAX_MERGES) { fclose(f); return -1; }
        if (bpe->merges_buf_used + (uint32_t)(alen + blen) > BPE_MERGES_BUF_BYTES) { fclose(f); return -1; }
        memcpy(bpe->merges_buf + bpe->merges_buf_used, line, (size_t)alen);
        bpe->merge_a[idx] = (BpeSpan){bpe->merges_buf_used, (uint32_t)alen};
        bpe->merges_buf_used += (uint32_t)alen;
        memcpy(bpe->merges_buf + bpe->merges_buf_used, space + 1, (size_t)blen);
        bpe->merge_b[idx] = (BpeSpan){bpe->merges_buf_used, (uint32_t)blen};
        bpe->merges_buf_used += (uint32_t)blen;

        char keybuf[512];
        if (alen + 1 + blen >= (int)sizeof keybuf) { fclose(f); return -1; }
        memcpy(keybuf, line, (size_t)alen);
        keybuf[alen] = '\0'; /* raw byte 0x00 separator: never produced by byte_enc (see header note) */
        memcpy(keybuf + alen + 1, space + 1, (size_t)blen);
        bpe_hash_insert(bpe->merge_hash, bpe->merges_buf, bpe->merge_a, idx, keybuf, (uint32_t)(alen + 1 + blen));
        bpe->merge_count++;
    }
    fclose(f);
    return 0;
}

static int gpt2_bpe_load(Gpt2Bpe *bpe, const char *vocab_path, const char *merges_path) {
    bpe_build_byte_table(bpe);
    if (bpe_load_vocab(bpe, vocab_path)) return -1;
    if (bpe_load_merges(bpe, merges_path)) return -1;
    return 0;
}

static int bpe_vocab_lookup(const Gpt2Bpe *bpe, const char *s, int len) {
    uint32_t h = bpe_fnv1a(s, (uint32_t)len) & (BPE_HASH_SLOTS - 1);
    for (;;) {
        int32_t id = bpe->vocab_hash[h];
        if (id < 0) return -1;
        BpeSpan sp = bpe->vocab_span[id];
        if (sp.len == (uint32_t)len && !memcmp(bpe->vocab_buf + sp.off, s, (size_t)len)) return id;
        h = (h + 1) & (BPE_HASH_SLOTS - 1);
    }
}

/* returns the rank of merging piece_a+piece_b if present in merges.txt, or INT32_MAX if absent */
static int32_t bpe_merge_rank(const Gpt2Bpe *bpe, const char *a, int alen, const char *b, int blen) {
    char keybuf[512];
    if (alen + 1 + blen >= (int)sizeof keybuf) return INT32_MAX;
    memcpy(keybuf, a, (size_t)alen);
    keybuf[alen] = '\0';
    memcpy(keybuf + alen + 1, b, (size_t)blen);
    uint32_t klen = (uint32_t)(alen + 1 + blen);
    uint32_t h = bpe_fnv1a(keybuf, klen) & (BPE_HASH_SLOTS - 1);
    for (;;) {
        int32_t idx = bpe->merge_hash[h];
        if (idx < 0) return INT32_MAX;
        BpeSpan sa = bpe->merge_a[idx], sb = bpe->merge_b[idx];
        if (sa.len == (uint32_t)alen && sb.len == (uint32_t)blen &&
            !memcmp(bpe->merges_buf + sa.off, a, (size_t)alen) && !memcmp(bpe->merges_buf + sb.off, b, (size_t)blen))
            return idx;
        h = (h + 1) & (BPE_HASH_SLOTS - 1);
    }
}

/* BPE-merges one pre-tokenized chunk (raw input bytes, tok_len of them) and
 * appends resulting token ids to ids_out (cap entries). Returns new count,
 * or -1 on overflow/vocab-miss (fail closed -- a merged piece with no
 * vocab entry indicates a loader/algorithm bug, not valid output). */
static int bpe_encode_chunk(const Gpt2Bpe *bpe, const char *tok, int tok_len, int *ids_out, int ids_cap, int ids_count) {
    static char work[BPE_MAX_TOKEN_PIECES * 2];
    BpeSpan pieces[BPE_MAX_TOKEN_PIECES];
    int n = 0, used = 0;
    for (int i = 0; i < tok_len; i++) {
        uint8_t b = (uint8_t)tok[i];
        uint8_t elen = bpe->byte_enc_len[b];
        if (n >= BPE_MAX_TOKEN_PIECES || used + elen > (int)sizeof work) return -1;
        memcpy(work + used, bpe->byte_enc[b], elen);
        pieces[n++] = (BpeSpan){(uint32_t)used, elen};
        used += elen;
    }
    if (n == 0) return ids_count;

    for (;;) {
        if (n == 1) break;
        int32_t best_rank = INT32_MAX;
        int best_i = -1;
        for (int i = 0; i < n - 1; i++) {
            int32_t r = bpe_merge_rank(bpe, work + pieces[i].off, (int)pieces[i].len, work + pieces[i + 1].off, (int)pieces[i + 1].len);
            if (r < best_rank) { best_rank = r; best_i = i; }
        }
        if (best_i < 0) break;

        /* Merge EVERY non-overlapping occurrence of the winning (piece_a,
         * piece_b) pair in this single left-to-right pass, not just the
         * first -- matching the real reference algorithm's `bpe()`
         * (which rebuilds the whole word in one pass per chosen bigram).
         * Merging only the first occurrence per iteration can diverge:
         * after that merge, a pair newly adjacent to it may score an
         * even lower rank and get picked next, before a second,
         * still-unmerged occurrence of the CURRENT winner ever gets its
         * turn -- caught by this file's differential check against the
         * real gpt2 merges.txt (case: " !@#$%^&*()_+-=[]{}"). */
        int win_alen = (int)pieces[best_i].len, win_blen = (int)pieces[best_i + 1].len;
        char win_a[BPE_MAX_TOKEN_PIECES * 2], win_b[BPE_MAX_TOKEN_PIECES * 2];
        memcpy(win_a, work + pieces[best_i].off, (size_t)win_alen);
        memcpy(win_b, work + pieces[best_i + 1].off, (size_t)win_blen);

        int write = 0, read = 0;
        while (read < n) {
            if (read < n - 1 && (int)pieces[read].len == win_alen && (int)pieces[read + 1].len == win_blen &&
                !memcmp(work + pieces[read].off, win_a, (size_t)win_alen) &&
                !memcmp(work + pieces[read + 1].off, win_b, (size_t)win_blen)) {
                pieces[write++] = (BpeSpan){pieces[read].off, pieces[read].len + pieces[read + 1].len};
                read += 2;
            } else {
                pieces[write++] = pieces[read];
                read += 1;
            }
        }
        n = write;
    }

    for (int i = 0; i < n; i++) {
        int id = bpe_vocab_lookup(bpe, work + pieces[i].off, (int)pieces[i].len);
        if (id < 0) return -1; /* fail closed: every merged piece must exist in vocab by construction */
        if (ids_count >= ids_cap) return -1;
        ids_out[ids_count++] = id;
    }
    return ids_count;
}

static int bpe_is_alpha(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
static int bpe_is_digit(char c) { return c >= '0' && c <= '9'; }
static int bpe_is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

/* ASCII-scoped pre-tokenizer (see file header for the exact scope
 * decision): splits text[0..len) into chunks per GPT-2's own regex
 * priority order, writing each chunk's [start,len) into chunks_out (cap
 * entries). Returns chunk count, or -1 on overflow. */
static int bpe_pretokenize(const char *text, int len, BpeSpan *chunks_out, int cap) {
    static const char *contractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    int n_chunks = 0;
    int i = 0;
    while (i < len) {
        int matched = 0;
        for (int c = 0; c < 7; c++) {
            int clen = (int)strlen(contractions[c]);
            if (i + clen <= len && !memcmp(text + i, contractions[c], (size_t)clen)) {
                if (n_chunks >= cap) return -1;
                chunks_out[n_chunks++] = (BpeSpan){(uint32_t)i, (uint32_t)clen};
                i += clen; matched = 1; break;
            }
        }
        if (matched) continue;

        int has_lead_space = (text[i] == ' ' && i + 1 < len);
        int j = i + (has_lead_space ? 1 : 0);
        if (j < len && bpe_is_alpha(text[j])) {
            int start = i, k = j;
            while (k < len && bpe_is_alpha(text[k])) k++;
            if (n_chunks >= cap) return -1;
            chunks_out[n_chunks++] = (BpeSpan){(uint32_t)start, (uint32_t)(k - start)};
            i = k; continue;
        }
        if (j < len && bpe_is_digit(text[j])) {
            int start = i, k = j;
            while (k < len && bpe_is_digit(text[k])) k++;
            if (n_chunks >= cap) return -1;
            chunks_out[n_chunks++] = (BpeSpan){(uint32_t)start, (uint32_t)(k - start)};
            i = k; continue;
        }
        if (j < len && !bpe_is_space(text[j])) {
            int start = i, k = j;
            while (k < len && !bpe_is_space(text[k]) && !bpe_is_alpha(text[k]) && !bpe_is_digit(text[k])) k++;
            if (n_chunks >= cap) return -1;
            chunks_out[n_chunks++] = (BpeSpan){(uint32_t)start, (uint32_t)(k - start)};
            i = k; continue;
        }

        /* whitespace run: \s+(?!\S) then \s+ fallback, characterized as:
         * k = maximal whitespace run length; if it reaches end of text,
         * consume all of it; else if k==1, consume the 1 char; else
         * consume k-1 chars (leaving exactly one for the next chunk's
         * optional leading-space prefix). */
        int start = i, k = i;
        while (k < len && bpe_is_space(text[k])) k++;
        int run = k - start;
        int take = (k == len) ? run : (run == 1 ? 1 : run - 1);
        if (take <= 0) { if (n_chunks >= cap) return -1; chunks_out[n_chunks++] = (BpeSpan){(uint32_t)start, 1}; i = start + 1; continue; }
        if (n_chunks >= cap) return -1;
        chunks_out[n_chunks++] = (BpeSpan){(uint32_t)start, (uint32_t)take};
        i = start + take;
    }
    return n_chunks;
}

/* Full encode: text -> token ids. Returns count, or -1 on overflow/error
 * (fail closed). */
static int gpt2_bpe_encode(const Gpt2Bpe *bpe, const char *text, int len, int *ids_out, int ids_cap) {
    static BpeSpan chunks[4096];
    int n_chunks = bpe_pretokenize(text, len, chunks, 4096);
    if (n_chunks < 0) return -1;
    int count = 0;
    for (int c = 0; c < n_chunks; c++) {
        count = bpe_encode_chunk(bpe, text + chunks[c].off, (int)chunks[c].len, ids_out, ids_cap, count);
        if (count < 0) return -1;
    }
    return count;
}

/* Full decode: token ids -> text bytes. Returns byte count written to
 * out (cap bytes), or -1 on overflow/unknown id. */
static int gpt2_bpe_decode(const Gpt2Bpe *bpe, const int *ids, int n_ids, char *out, int out_cap) {
    /* byte_decoder: map each byte_enc[] UTF-8 sequence back to its
     * original byte. Only 256 entries; linear scan is fine (called once
     * per decode, not per byte). */
    int n = 0;
    for (int t = 0; t < n_ids; t++) {
        int id = ids[t];
        if (id < 0 || id >= bpe->vocab_count) return -1;
        BpeSpan sp = bpe->vocab_span[id];
        const char *piece = bpe->vocab_buf + sp.off;
        uint32_t plen = sp.len;
        uint32_t pi = 0;
        while (pi < plen) {
            int found = -1;
            for (int b = 0; b < 256; b++) {
                uint8_t el = bpe->byte_enc_len[b];
                if (pi + el <= plen && !memcmp(piece + pi, bpe->byte_enc[b], el)) { found = b; pi += el; break; }
            }
            if (found < 0) return -1;
            if (n >= out_cap) return -1;
            out[n++] = (char)found;
        }
    }
    return n;
}

#endif
