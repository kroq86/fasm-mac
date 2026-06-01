; Binary tree helpers for LeetCode-style TreeNode tasks.

struc TreeNode
{
	.val dq ?
	.left dq ?
	.right dq ?
}

virtual at 0
	_TreeNode TreeNode
	TREE_NODE_SIZE = $
end virtual

TREE_NODE_VAL_OFF equ 0
TREE_NODE_LEFT_OFF equ 8
TREE_NODE_RIGHT_OFF equ 16

; rdi = TreeNode*
; rax = maximum depth
tree_max_depth:
	test	rdi, rdi
	jz	.tmd_zero
	push	r12
	push	rdi
	mov	rdi, [rdi + TREE_NODE_LEFT_OFF]
	call	tree_max_depth
	mov	r12, rax
	pop	rdi
	push	r12
	mov	rdi, [rdi + TREE_NODE_RIGHT_OFF]
	call	tree_max_depth
	pop	r12
	cmp	r12, rax
	jge	.tmd_left_max
	inc	rax
	pop	r12
	ret
.tmd_left_max:
	mov	rax, r12
	inc	rax
	pop	r12
	ret
.tmd_zero:
	xor	rax, rax
	ret

; rdi = TreeNode*
; rax = original root after in-place inversion
tree_invert:
	test	rdi, rdi
	jz	.ti_done
	push	r12
	mov	r12, rdi
	mov	rax, [r12 + TREE_NODE_LEFT_OFF]
	mov	rcx, [r12 + TREE_NODE_RIGHT_OFF]
	mov	[r12 + TREE_NODE_LEFT_OFF], rcx
	mov	[r12 + TREE_NODE_RIGHT_OFF], rax
	mov	rdi, [r12 + TREE_NODE_LEFT_OFF]
	call	tree_invert
	mov	rdi, [r12 + TREE_NODE_RIGHT_OFF]
	call	tree_invert
	mov	rax, r12
	pop	r12
	ret
.ti_done:
	xor	rax, rax
	ret

; rdi = TreeNode*
; Prints values in preorder with trailing spaces. Requires print_int_sp.
tree_print_preorder:
	test	rdi, rdi
	jz	.tpp_done
	push	r12
	mov	r12, rdi
	mov	rax, [r12 + TREE_NODE_VAL_OFF]
	call	print_int_sp
	mov	rdi, [r12 + TREE_NODE_LEFT_OFF]
	call	tree_print_preorder
	mov	rdi, [r12 + TREE_NODE_RIGHT_OFF]
	call	tree_print_preorder
	pop	r12
.tpp_done:
	ret
