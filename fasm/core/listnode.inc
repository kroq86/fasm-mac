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

; rdi = sorted list A, rsi = sorted list B, rdx = dummy ListNode*
; rax = merged sorted list head
list_merge_sorted:
	mov	qword [rdx + LIST_NODE_NEXT_OFF], 0
	mov	r8, rdx
.lms_loop:
	test	rdi, rdi
	jz	.lms_take_b_rest
	test	rsi, rsi
	jz	.lms_take_a_rest
	mov	rax, [rdi + LIST_NODE_VAL_OFF]
	cmp	rax, [rsi + LIST_NODE_VAL_OFF]
	jg	.lms_take_b
.lms_take_a:
	mov	[r8 + LIST_NODE_NEXT_OFF], rdi
	mov	r8, rdi
	mov	rdi, [rdi + LIST_NODE_NEXT_OFF]
	jmp	.lms_loop
.lms_take_b:
	mov	[r8 + LIST_NODE_NEXT_OFF], rsi
	mov	r8, rsi
	mov	rsi, [rsi + LIST_NODE_NEXT_OFF]
	jmp	.lms_loop
.lms_take_a_rest:
	mov	[r8 + LIST_NODE_NEXT_OFF], rdi
	jmp	.lms_done
.lms_take_b_rest:
	mov	[r8 + LIST_NODE_NEXT_OFF], rsi
.lms_done:
	mov	rax, [rdx + LIST_NODE_NEXT_OFF]
	ret

; rdi = head ListNode*
; rax = middle node
list_middle:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rdi
.lm_loop:
	test	r13, r13
	jz	.lm_done
	mov	r13, [r13 + LIST_NODE_NEXT_OFF]
	test	r13, r13
	jz	.lm_done
	mov	r12, [r12 + LIST_NODE_NEXT_OFF]
	mov	r13, [r13 + LIST_NODE_NEXT_OFF]
	jmp	.lm_loop
.lm_done:
	mov	rax, r12
	pop	r13
	pop	r12
	ret

; rdi = head ListNode*
; rax = 1 if a cycle exists, 0 otherwise
list_has_cycle:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rdi
.lhc_loop:
	test	r13, r13
	jz	.lhc_no
	mov	r13, [r13 + LIST_NODE_NEXT_OFF]
	test	r13, r13
	jz	.lhc_no
	mov	r12, [r12 + LIST_NODE_NEXT_OFF]
	mov	r13, [r13 + LIST_NODE_NEXT_OFF]
	cmp	r12, r13
	je	.lhc_yes
	jmp	.lhc_loop
.lhc_yes:
	mov	rax, 1
	pop	r13
	pop	r12
	ret
.lhc_no:
	xor	rax, rax
	pop	r13
	pop	r12
	ret

; rdi = head ListNode*, rsi = int64 scratch buffer, rdx = buffer capacity
; rax = 1 if palindrome, 0 otherwise
list_is_palindrome:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	rbx, rdi
	xor	r12, r12
.lip_load:
	test	rbx, rbx
	jz	.lip_compare
	cmp	r12, rdx
	jae	.lip_no
	mov	rax, [rbx + LIST_NODE_VAL_OFF]
	mov	[rsi + r12 * 8], rax
	inc	r12
	mov	rbx, [rbx + LIST_NODE_NEXT_OFF]
	jmp	.lip_load
.lip_compare:
	xor	r13, r13
	mov	r14, r12
	test	r14, r14
	jz	.lip_yes
	dec	r14
.lip_cmp_loop:
	cmp	r13, r14
	jae	.lip_yes
	mov	rax, [rsi + r13 * 8]
	cmp	rax, [rsi + r14 * 8]
	jne	.lip_no
	inc	r13
	dec	r14
	jmp	.lip_cmp_loop
.lip_yes:
	mov	rax, 1
	jmp	.lip_done
.lip_no:
	xor	rax, rax
.lip_done:
	pop	r14
	pop	r13
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
