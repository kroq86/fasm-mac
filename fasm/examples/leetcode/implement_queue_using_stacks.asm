; LeetCode 232. Implement Queue using Stacks
; https://leetcode.com/problems/implement-queue-using-stacks/
; push 1, push 2, peek, pop, empty  =>  1 1 0

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"
include "fasm/core/stack.inc"

entry start

start:
	lea	rdi, [queue]
	lea	rsi, [stack_in]
	lea	rdx, [stack_out]
	mov	rcx, queue_cap
	call	int_queue2stack_init
	lea	rdi, [queue]
	mov	rsi, 1
	call	int_queue2stack_push
	lea	rdi, [queue]
	mov	rsi, 2
	call	int_queue2stack_push
	lea	rdi, [queue]
	call	int_queue2stack_peek
	call	print_int_sp
	lea	rdi, [queue]
	call	int_queue2stack_pop
	call	print_int_sp
	lea	rdi, [queue]
	call	int_queue2stack_empty
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

stack_in rq 8
stack_out rq 8
queue rb INT_QUEUE2STACK_SIZE
queue_cap = 8

include "fasm/core/runtime_bss.inc"
runtime_print_bss
