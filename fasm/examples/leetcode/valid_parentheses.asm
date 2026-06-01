; LeetCode 20. Valid Parentheses
; https://leetcode.com/problems/valid-parentheses/
; s="()[]{}"  =>  1

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"
include "fasm/core/stack.inc"

entry start

start:
	lea	rdi, [s]
	lea	rsi, [paren_stack]
	mov	rdx, paren_stack_cap
	call	valid_parentheses_core
	call	print_bool_nl
	exit EXIT_SUCCESS

segment readable writeable

s db '()[]{}', 0
paren_stack rb 32
paren_stack_cap = 32

include "fasm/core/runtime_bss.inc"
runtime_print_bss
