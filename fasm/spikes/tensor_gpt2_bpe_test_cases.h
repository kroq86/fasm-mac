/* Auto-generated from an independent Python oracle -- see
 * fasm/spikes/tensor_gpt2_bpe_differential_check.c's header comment for
 * the exact generating command. Do not hand-edit. */
typedef struct { const char *text; int ids[64]; int n; } BpeTestCase;
static const BpeTestCase BPE_TEST_CASES[] = {
    {"Hello, world!", {15496, 11, 995, 0}, 4},
    {"Hello, world! This is a test 123.", {15496, 11, 995, 0, 770, 318, 257, 1332, 17031, 13}, 10},
    {"  double  spaces\tand\ttabs", {220, 4274, 220, 9029, 197, 392, 197, 8658, 82}, 9},
    {"leading space test", {12294, 2272, 1332}, 3},
    {" leading space test", {3756, 2272, 1332}, 3},
    {"trailing space test ", {9535, 4386, 2272, 1332, 220}, 5},
    {"don't can't won't I'll you're we've they'd I'm", {9099, 470, 460, 470, 1839, 470, 314, 1183, 345, 821, 356, 1053, 484, 1549, 314, 1101}, 16},
    {"UPPERCASE lowercase MiXeD", {8577, 18973, 34, 11159, 2793, 7442, 13756, 55, 68, 35}, 10},
    {"numbers 0 1 22 333 4444", {77, 17024, 657, 352, 2534, 23460, 604, 30272}, 8},
    {"punctuation: !@#$%^&*()_+-=[]{}", {79, 16260, 2288, 25, 5145, 31, 29953, 4, 61, 5, 9, 3419, 62, 10, 12, 28, 21737, 90, 92}, 19},
    {"a", {64}, 1},
    {"", {0}, 0},
    {"   ", {220, 220, 220}, 3},
    {"a  b", {64, 220, 275}, 3},
    {"a b", {64, 275}, 2},
    {"a\tb\nc", {64, 197, 65, 198, 66}, 5},
    {"The quick brown fox jumps over the lazy dog.", {464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13}, 10},
    {"repeated repeated repeated", {45956, 515, 5100, 5100}, 4},
    {"<|endoftext|>", {27, 91, 437, 1659, 5239, 91, 29}, 7},
    {"multiple   spaces   between   words", {48101, 220, 220, 9029, 220, 220, 1022, 220, 220, 2456}, 10},
    {"newline\nseparated\nwords", {3605, 1370, 198, 25512, 515, 198, 10879}, 7},
};
enum { BPE_TEST_CASE_COUNT = 21 };
