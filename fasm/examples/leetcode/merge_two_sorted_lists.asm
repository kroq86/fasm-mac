; LeetCode 21. Merge Two Sorted Lists
; https://leetcode.com/problems/merge-two-sorted-lists/
; list1=1->2->4, list2=1->3->4  =>  1 1 2 3 4 4

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/dynvec.inc"
include "fasm/examples/leetcode/common.inc"
include "fasm/core/listnode.inc"
include "fasm/core/leetcode.inc"

entry start

start:
	lea	rdi, [a1]
	lea	rsi, [b1]
	lea	rdx, [dummy]
	call	list_merge_sorted
	mov	rdi, rax
	call	list_print
	mov	al, 10
	call	print_char
	exit EXIT_SUCCESS

segment readable writeable

a1 dq 1, a2
a2 dq 2, a3
a3 dq 4, 0
b1 dq 1, b2
b2 dq 3, b3
b3 dq 4, 0
dummy dq 0, 0

include "fasm/core/runtime_bss.inc"
runtime_print_bss
