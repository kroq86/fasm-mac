; LeetCode 141. Linked List Cycle
; https://leetcode.com/problems/linked-list-cycle/
; head=3->2->0->-4, tail connects to 2  =>  1

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
	call	list_has_cycle
	call	print_bool_nl
	exit EXIT_SUCCESS

segment readable writeable

n1 dq 3, n2
n2 dq 2, n3
n3 dq 0, n4
n4 dq -4, n2

include "fasm/core/runtime_bss.inc"
runtime_print_bss
