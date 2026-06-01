; Fixed-buffer stack and two-stack queue helpers.

struc ByteStack
{
	.data dq ?
	.count dq ?
	.cap dq ?
}

struc IntQueue2Stack
{
	.in_data dq ?
	.out_data dq ?
	.in_count dq ?
	.out_count dq ?
	.cap dq ?
}

virtual at 0
	_ByteStack ByteStack
	BYTE_STACK_SIZE = $
	_IntQueue2Stack IntQueue2Stack
	INT_QUEUE2STACK_SIZE = $
end virtual

BYTE_STACK_DATA_OFF equ 0
BYTE_STACK_COUNT_OFF equ 8
BYTE_STACK_CAP_OFF equ 16

INT_Q2S_IN_DATA_OFF equ 0
INT_Q2S_OUT_DATA_OFF equ 8
INT_Q2S_IN_COUNT_OFF equ 16
INT_Q2S_OUT_COUNT_OFF equ 24
INT_Q2S_CAP_OFF equ 32

; rdi = ByteStack*, rsi = byte buffer, rdx = capacity
byte_stack_init:
	mov	[rdi + BYTE_STACK_DATA_OFF], rsi
	mov	qword [rdi + BYTE_STACK_COUNT_OFF], 0
	mov	[rdi + BYTE_STACK_CAP_OFF], rdx
	ret

; rdi = ByteStack*, sil = byte
; rax = 1 on success, 0 if full
byte_stack_push:
	mov	rax, [rdi + BYTE_STACK_COUNT_OFF]
	cmp	rax, [rdi + BYTE_STACK_CAP_OFF]
	jae	.bsp_full
	mov	rcx, [rdi + BYTE_STACK_DATA_OFF]
	mov	[rcx + rax], sil
	inc	qword [rdi + BYTE_STACK_COUNT_OFF]
	mov	rax, 1
	ret
.bsp_full:
	xor	rax, rax
	ret

; rdi = ByteStack*
; rax = popped byte, or -1 if empty
byte_stack_pop:
	mov	rax, [rdi + BYTE_STACK_COUNT_OFF]
	test	rax, rax
	jz	.bsp_empty
	dec	rax
	mov	[rdi + BYTE_STACK_COUNT_OFF], rax
	mov	rcx, [rdi + BYTE_STACK_DATA_OFF]
	movzx	rax, byte [rcx + rax]
	ret
.bsp_empty:
	mov	rax, -1
	ret

; rdi = ByteStack*
; rax = 1 if empty
byte_stack_empty:
	mov	rax, [rdi + BYTE_STACK_COUNT_OFF]
	test	rax, rax
	setz	al
	movzx	rax, al
	ret

; rdi = null-terminated string, rsi = byte buffer, rdx = buffer capacity
; rax = 1 if parentheses are valid, 0 otherwise.
valid_parentheses_core:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	xor	r15, r15
	xor	rbx, rbx
.vpc_loop:
	mov	al, [r12 + rbx]
	test	al, al
	jz	.vpc_end
	cmp	al, '('
	je	.vpc_push
	cmp	al, '['
	je	.vpc_push
	cmp	al, '{'
	je	.vpc_push
	mov	r11b, al
	test	r15, r15
	jz	.vpc_no
	dec	r15
	mov	al, [r13 + r15]
	cmp	r11b, ')'
	je	.vpc_close_paren
	cmp	r11b, ']'
	je	.vpc_close_bracket
	cmp	al, '{'
	je	.vpc_next
	jmp	.vpc_no
.vpc_close_paren:
	cmp	al, '('
	je	.vpc_next
	jmp	.vpc_no
.vpc_close_bracket:
	cmp	al, '['
	je	.vpc_next
	jmp	.vpc_no
.vpc_push:
	cmp	r15, r14
	jae	.vpc_no
	mov	[r13 + r15], al
	inc	r15
.vpc_next:
	inc	rbx
	jmp	.vpc_loop
.vpc_end:
	test	r15, r15
	setz	al
	movzx	rax, al
	jmp	.vpc_done
.vpc_no:
	xor	rax, rax
.vpc_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = queue*, rsi = input stack buffer, rdx = output stack buffer, rcx = capacity
int_queue2stack_init:
	mov	[rdi + INT_Q2S_IN_DATA_OFF], rsi
	mov	[rdi + INT_Q2S_OUT_DATA_OFF], rdx
	mov	qword [rdi + INT_Q2S_IN_COUNT_OFF], 0
	mov	qword [rdi + INT_Q2S_OUT_COUNT_OFF], 0
	mov	[rdi + INT_Q2S_CAP_OFF], rcx
	ret

; rdi = queue*, rsi = value
; rax = 1 on success, 0 if full
int_queue2stack_push:
	mov	rax, [rdi + INT_Q2S_IN_COUNT_OFF]
	mov	rcx, [rdi + INT_Q2S_OUT_COUNT_OFF]
	add	rcx, rax
	cmp	rcx, [rdi + INT_Q2S_CAP_OFF]
	jae	.iq2sp_full
	mov	rcx, [rdi + INT_Q2S_IN_DATA_OFF]
	mov	[rcx + rax * 8], rsi
	inc	qword [rdi + INT_Q2S_IN_COUNT_OFF]
	mov	rax, 1
	ret
.iq2sp_full:
	xor	rax, rax
	ret

; rdi = queue*
int_queue2stack_move:
	cmp	qword [rdi + INT_Q2S_OUT_COUNT_OFF], 0
	jne	.iq2sm_done
.iq2sm_loop:
	cmp	qword [rdi + INT_Q2S_IN_COUNT_OFF], 0
	je	.iq2sm_done
	dec	qword [rdi + INT_Q2S_IN_COUNT_OFF]
	mov	rax, [rdi + INT_Q2S_IN_COUNT_OFF]
	mov	rcx, [rdi + INT_Q2S_IN_DATA_OFF]
	mov	r8, [rcx + rax * 8]
	mov	rax, [rdi + INT_Q2S_OUT_COUNT_OFF]
	mov	rcx, [rdi + INT_Q2S_OUT_DATA_OFF]
	mov	[rcx + rax * 8], r8
	inc	qword [rdi + INT_Q2S_OUT_COUNT_OFF]
	jmp	.iq2sm_loop
.iq2sm_done:
	ret

; rdi = queue*
; rax = front value, or -1 if empty
int_queue2stack_peek:
	push	rdi
	call	int_queue2stack_move
	pop	rdi
	mov	rax, [rdi + INT_Q2S_OUT_COUNT_OFF]
	test	rax, rax
	jz	.iq2spe_empty
	dec	rax
	mov	rcx, [rdi + INT_Q2S_OUT_DATA_OFF]
	mov	rax, [rcx + rax * 8]
	ret
.iq2spe_empty:
	mov	rax, -1
	ret

; rdi = queue*
; rax = popped value, or -1 if empty
int_queue2stack_pop:
	push	rdi
	call	int_queue2stack_peek
	pop	rdi
	cmp	rax, -1
	je	.iq2spo_done
	dec	qword [rdi + INT_Q2S_OUT_COUNT_OFF]
.iq2spo_done:
	ret

; rdi = queue*
; rax = 1 if empty
int_queue2stack_empty:
	mov	rax, [rdi + INT_Q2S_IN_COUNT_OFF]
	add	rax, [rdi + INT_Q2S_OUT_COUNT_OFF]
	test	rax, rax
	setz	al
	movzx	rax, al
	ret
