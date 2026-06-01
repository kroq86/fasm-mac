; LeetCode 234. Palindrome Linked List
; https://leetcode.com/problems/palindrome-linked-list/
; head=1->2->2->1  =>  1

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
	lea	rdi, [n1]
	lea	rsi, [vals]
	mov	rdx, vals_cap
	call	list_is_palindrome
	call	print_bool_nl
	exit EXIT_SUCCESS

segment readable writeable

n1 dq 1, n2
n2 dq 2, n3
n3 dq 2, n4
n4 dq 1, 0
vals rq 8
vals_cap = 8

include "fasm/core/runtime_bss.inc"
runtime_print_bss
