; LeetCode 876. Middle of the Linked List
; https://leetcode.com/problems/middle-of-the-linked-list/
; head=1->2->3->4->5  =>  3

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
	call	list_middle
	mov	rax, [rax + LIST_NODE_VAL_OFF]
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

n1 dq 1, n2
n2 dq 2, n3
n3 dq 3, n4
n4 dq 4, n5
n5 dq 5, 0

include "fasm/core/runtime_bss.inc"
runtime_print_bss
