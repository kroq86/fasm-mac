#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fasm-leetcode.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

expected_for() {
  case "$1" in
    best_time_to_buy_sell_stock) printf '%s' '5' ;;
    binary_search) printf '%s' '4' ;;
    climbing_stairs) printf '%s' '8' ;;
    contains_duplicate) printf '%s' '1' ;;
    first_unique_character) printf '%s' '0' ;;
    house_robber) printf '%s' '4' ;;
    implement_queue_using_stacks) printf '%s' '1 1 0' ;;
    intersection_of_two_arrays) printf '%s' '2 ' ;;
    invert_binary_tree) printf '%s' '4 7 9 6 2 3 1 ' ;;
    linked_list_cycle) printf '%s' '1' ;;
    majority_element) printf '%s' '2' ;;
    maximum_depth_binary_tree) printf '%s' '3' ;;
    maximum_subarray) printf '%s' '6' ;;
    merge_sorted_array) printf '%s' '1 2 2 3 5 6 ' ;;
    merge_two_sorted_lists) printf '%s' '1 1 2 3 4 4 ' ;;
    middle_of_linked_list) printf '%s' '3' ;;
    missing_number) printf '%s' '2' ;;
    move_zeroes) printf '%s' '1 3 12 0 0 ' ;;
    nested_list_weight_sum) printf '%s' '10' ;;
    number_of_islands) printf '%s' '1' ;;
    palindrome_linked_list) printf '%s' '1' ;;
    remove_duplicates_sorted_array) printf '%s' '5 0 1 2 3 4 ' ;;
    reverse_linked_list) printf '%s' '5 4 3 2 1 ' ;;
    search_insert_position) printf '%s' '2' ;;
    single_number) printf '%s' '4' ;;
    sort_array) printf '%s' '-1 0 1 2 3 ' ;;
    two_sum) printf '%s' '0 1' ;;
    two_sum_hashmap) printf '%s' '0 1' ;;
    valid_anagram) printf '%s' '1' ;;
    valid_parentheses) printf '%s' '1' ;;
    *) return 1 ;;
  esac
}

shopt -s nullglob
for asm in "$ROOT"/fasm/examples/leetcode/*.asm; do
  name="$(basename "$asm" .asm)"
  if ! expected="$(expected_for "$name")"; then
    echo "missing expected output for $name" >&2
    exit 1
  fi

  bin="$OUT_DIR/$name"
  fasm "$asm" "$bin" >/dev/null
  actual="$(arch -x86_64 "$bin")"

  if [[ "$actual" != "$expected" ]]; then
    printf 'FAIL %s\nexpected: %q\nactual:   %q\n' "$name" "$expected" "$actual" >&2
    exit 1
  fi
  printf 'ok %s\n' "$name"
done

echo "all LeetCode examples passed"
