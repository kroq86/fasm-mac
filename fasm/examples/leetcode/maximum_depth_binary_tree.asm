; LeetCode 104. Maximum Depth of Binary Tree
; https://leetcode.com/problems/maximum-depth-of-binary-tree/
; tree=[3,9,20,null,null,15,7]  =>  3

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/examples/leetcode/common.inc"
include "fasm/core/tree.inc"

entry start

start:
	lea	rdi, [root]
	call	tree_max_depth
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

root dq 3, n9, n20
n9 dq 9, 0, 0
n20 dq 20, n15, n7
n15 dq 15, 0, 0
n7 dq 7, 0, 0

include "fasm/core/runtime_bss.inc"
runtime_print_bss
