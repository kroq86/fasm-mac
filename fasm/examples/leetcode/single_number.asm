; LeetCode 136. Single Number
; https://leetcode.com/problems/single-number/
; nums=[4,1,2,1,2]  =>  4

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"

entry start

start:
	xor	rbx, rbx
	xor	rax, rax
.loop:
	cmp	rbx, test_nums_count
	jae	.done
	xor	rax, [test_nums + rbx * 8]
	inc	rbx
	jmp	.loop
.done:
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

test_nums dq 4, 1, 2, 1, 2
test_nums_count = 5

include "fasm/core/runtime_bss.inc"
runtime_print_bss
