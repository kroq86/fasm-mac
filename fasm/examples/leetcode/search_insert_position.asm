; LeetCode 35. Search Insert Position
; https://leetcode.com/problems/search-insert-position/
; nums=[1,3,5,6], target=5  =>  2

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"

entry start

start:
	xor	r12, r12
	mov	r13, nums_count
.loop:
	cmp	r12, r13
	jae	.done
	mov	rax, r12
	add	rax, r13
	shr	rax, 1
	mov	rbx, rax
	mov	rcx, [nums + rbx * 8]
	cmp	rcx, target
	jl	.right
	mov	r13, rbx
	jmp	.loop
.right:
	lea	r12, [rbx + 1]
	jmp	.loop
.done:
	mov	rax, r12
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

nums dq 1, 3, 5, 6
nums_count = 4
target = 5

include "fasm/core/runtime_bss.inc"
runtime_print_bss
