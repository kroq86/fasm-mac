; Singly linked list nodes (LeetCode ListNode style).

struc ListNode
{
	.val dq ?
	.next dq ?
}

virtual at 0
	_ListNode ListNode
	LIST_NODE_SIZE = $
end virtual

LIST_NODE_VAL_OFF equ 0
LIST_NODE_NEXT_OFF equ 8

; rdi = Heap*, rsi = val
; rax = ListNode*
listnode_new:
	push	r12
	mov	r12, rsi
	mov	rsi, LIST_NODE_SIZE
	call	heap_alloc
	mov	[rax + LIST_NODE_VAL_OFF], r12
	mov	qword [rax + LIST_NODE_NEXT_OFF], 0
	pop	r12
	ret

; rdi = Heap*, rsi = head (ListNode*), rdx = val
; rax = head
list_push_back:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	rbx, rsi
	mov	r13, rdx
	test	rbx, rbx
	jnz	.lpb_find_tail
	mov	rdi, r12
	mov	rsi, r13
	call	listnode_new
	jmp	.lpb_done
.lpb_find_tail:
	mov	rax, rbx
.lpb_walk:
	mov	rcx, [rax + LIST_NODE_NEXT_OFF]
	test	rcx, rcx
	jz	.lpb_append
	mov	rax, rcx
	jmp	.lpb_walk
.lpb_append:
	mov	rdi, r12
	mov	rsi, r13
	push	rax
	call	listnode_new
	pop	r10
	mov	[r10 + LIST_NODE_NEXT_OFF], rax
	mov	rax, rbx
.lpb_done:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = Heap*, rsi = VecHeader* (VAL_INT elements)
; rax = ListNode* head
list_build_from_vec:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	rbx, rsi
	xor	r13, r13
	xor	r14, r14
.lbfv_loop:
	mov	rcx, [rbx + VEC_COUNT_OFF]
	cmp	r14, rcx
	jae	.lbfv_done
	mov	rdi, rbx
	mov	rsi, r14
	call	vec_get_int
	mov	rdx, rax
	mov	rdi, r12
	mov	rsi, r13
	call	list_push_back
	mov	r13, rax
	inc	r14
	jmp	.lbfv_loop
.lbfv_done:
	mov	rax, r13
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = head ListNode*
; rax = new head (reversed in place)
list_reverse:
	push	rbx
	push	r12
	mov	r12, rdi
	xor	rbx, rbx
.lr_loop:
	test	r12, r12
	jz	.lr_done
	mov	rax, [r12 + LIST_NODE_NEXT_OFF]
	mov	[r12 + LIST_NODE_NEXT_OFF], rbx
	mov	rbx, r12
	mov	r12, rax
	jmp	.lr_loop
.lr_done:
	mov	rax, rbx
	pop	r12
	pop	rbx
	ret

; rdi = head ListNode*
list_print:
	push	rbx
	mov	rbx, rdi
.lp_loop:
	test	rbx, rbx
	jz	.lp_done
	mov	rax, [rbx + LIST_NODE_VAL_OFF]
	call	print_int_sp
	mov	rbx, [rbx + LIST_NODE_NEXT_OFF]
	jmp	.lp_loop
.lp_done:
	pop	rbx
	ret

; rdi = head ListNode*
gc_mark_list:
	push	rbx
	mov	rbx, rdi
.gml_loop:
	test	rbx, rbx
	jz	.gml_done
	mark_block rbx
	mov	rbx, [rbx + LIST_NODE_NEXT_OFF]
	jmp	.gml_loop
.gml_done:
	pop	rbx
	ret
