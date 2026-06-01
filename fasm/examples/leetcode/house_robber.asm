; LeetCode 198. House Robber
; https://leetcode.com/problems/house-robber/
; nums=[1,2,3,1]  =>  4

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"
include "fasm/core/dp.inc"

entry start

start:
	lea	rdi, [nums]
	mov	rsi, nums_count
	call	dp_house_robber
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

nums dq 1, 2, 3, 1
nums_count = 4

include "fasm/core/runtime_bss.inc"
runtime_print_bss
