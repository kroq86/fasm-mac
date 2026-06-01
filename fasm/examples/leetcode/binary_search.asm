; LeetCode 704. Binary Search
; https://leetcode.com/problems/binary-search/
; nums=[-1,0,3,5,9,12], target=9  =>  4

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"

entry start

start:
	xor	r12, r12
	mov	r13, nums_count - 1
.loop:
	cmp	r12, r13
	ja	.not_found
	mov	rax, r12
	add	rax, r13
	shr	rax, 1
	mov	rbx, rax
	mov	rcx, [nums + rbx * 8]
	cmp	rcx, target
	je	.found
	jl	.go_right
	lea	r13, [rbx - 1]
	jmp	.loop
.go_right:
	lea	r12, [rbx + 1]
	jmp	.loop
.found:
	mov	rax, rbx
	jmp	.print
.not_found:
	mov	rax, -1
.print:
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

nums dq -1, 0, 3, 5, 9, 12
nums_count = 6
target = 9

include "fasm/core/runtime_bss.inc"
runtime_print_bss
