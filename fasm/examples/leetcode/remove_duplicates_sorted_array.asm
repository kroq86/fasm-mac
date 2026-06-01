; LeetCode 26. Remove Duplicates from Sorted Array
; https://leetcode.com/problems/remove-duplicates-from-sorted-array/
; nums=[0,0,1,1,1,2,2,3,3,4]  =>  5 0 1 2 3 4

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"

entry start

start:
	mov	r12, 1
	mov	r13, 1
.loop:
	cmp	r13, nums_count
	jae	.done
	mov	rax, [nums + r13 * 8]
	mov	rcx, [nums + r13 * 8 - 8]
	cmp	rax, rcx
	je	.next
	mov	[nums + r12 * 8], rax
	inc	r12
.next:
	inc	r13
	jmp	.loop
.done:
	mov	rax, r12
	call	print_int_sp
	lea	rdi, [nums]
	mov	rsi, r12
	call	print_i64_array
	exit EXIT_SUCCESS

segment readable writeable

nums dq 0, 0, 1, 1, 1, 2, 2, 3, 3, 4
nums_count = 10

include "fasm/core/runtime_bss.inc"
runtime_print_bss
