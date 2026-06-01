; LeetCode 226. Invert Binary Tree
; https://leetcode.com/problems/invert-binary-tree/
; preorder after invert of [4,2,7,1,3,6,9]  =>  4 7 9 6 2 3 1

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"
include "fasm/core/tree.inc"

entry start

start:
	lea	rdi, [root]
	call	tree_invert
	lea	rdi, [root]
	call	tree_print_preorder
	mov	al, 10
	call	print_char
	exit EXIT_SUCCESS

segment readable writeable

root dq 4, n2, n7
n2 dq 2, n1, n3
n7 dq 7, n6, n9
n1 dq 1, 0, 0
n3 dq 3, 0, 0
n6 dq 6, 0, 0
n9 dq 9, 0, 0

include "fasm/core/runtime_bss.inc"
runtime_print_bss
