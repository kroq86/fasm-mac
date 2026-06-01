; LeetCode 283. Move Zeroes
; https://leetcode.com/problems/move-zeroes/
; nums=[0,1,0,3,12]  =>  1 3 12 0 0

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"

entry start

start:
	xor	r12, r12
	xor	r13, r13
.compact:
	cmp	r13, nums_count
	jae	.fill_zeroes
	mov	rax, [nums + r13 * 8]
	test	rax, rax
	jz	.next
	mov	[nums + r12 * 8], rax
	inc	r12
.next:
	inc	r13
	jmp	.compact
.fill_zeroes:
	cmp	r12, nums_count
	jae	.print
	mov	qword [nums + r12 * 8], 0
	inc	r12
	jmp	.fill_zeroes
.print:
	lea	rdi, [nums]
	mov	rsi, nums_count
	call	print_i64_array
	exit EXIT_SUCCESS

segment readable writeable

nums dq 0, 1, 0, 3, 12
nums_count = 5

include "fasm/core/runtime_bss.inc"
runtime_print_bss
