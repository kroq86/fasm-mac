; LeetCode 200. Number of Islands
; https://leetcode.com/problems/number-of-islands/
; grid sample with one island  =>  1

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"
include "fasm/core/grid.inc"

ROWS = 4
COLS = 5

entry start

start:
	lea	rdi, [grid]
	mov	rsi, ROWS
	mov	rdx, COLS
	mov	rcx, '1'
	mov	r8, '0'
	call	grid_count_components4
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

grid db '1','1','1','1','0'
     db '1','1','0','1','0'
     db '1','1','0','0','0'
     db '0','0','0','0','0'

include "fasm/core/runtime_bss.inc"
runtime_print_bss
grid_bss
