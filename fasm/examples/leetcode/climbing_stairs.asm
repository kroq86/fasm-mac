; LeetCode 70. Climbing Stairs
; https://leetcode.com/problems/climbing-stairs/
; n=5  =>  8

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"
include "fasm/core/dp.inc"

entry start

start:
	mov	rdi, n
	call	dp_climbing_stairs
	call	print_int_nl
	exit EXIT_SUCCESS

n = 5

segment readable writeable

include "fasm/core/runtime_bss.inc"
runtime_print_bss
