; LeetCode 53. Maximum Subarray
; https://leetcode.com/problems/maximum-subarray/
; nums=[-2,1,-3,4,-1,2,1,-5,4]  =>  6

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"
include "fasm/core/dp.inc"

entry start

start:
	lea	rdi, [nums]
	mov	rsi, nums_count
	call	dp_max_subarray
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

nums dq -2, 1, -3, 4, -1, 2, 1, -5, 4
nums_count = 9

include "fasm/core/runtime_bss.inc"
runtime_print_bss
