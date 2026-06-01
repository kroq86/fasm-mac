; LeetCode 268. Missing Number
; https://leetcode.com/problems/missing-number/
; nums=[3,0,1]  =>  2

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"

entry start

start:
	mov	rax, test_nums_count
	xor	rbx, rbx
.loop:
	cmp	rbx, test_nums_count
	jae	.done
	xor	rax, rbx
	xor	rax, [test_nums + rbx * 8]
	inc	rbx
	jmp	.loop
.done:
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

test_nums dq 3, 0, 1
test_nums_count = 3

include "fasm/core/runtime_bss.inc"
runtime_print_bss
