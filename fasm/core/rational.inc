; Rational numbers backed by two signed int64 fields.
;
; Invariant after rational_set/rational_*:
; - denominator is positive
; - numerator and denominator are reduced by gcd(abs(num), den)
; - zero is represented as 0/1

struc Rational
{
	.num dq ?
	.den dq ?
}

virtual at 0
	_Rational Rational
	RATIONAL_SIZE = $
end virtual

RATIONAL_NUM_OFF equ 0
RATIONAL_DEN_OFF equ 8

RATIONAL_OK equ 0
RATIONAL_ERR_ZERO_DEN equ -1
RATIONAL_ERR_PARSE equ -2

; rdi = signed int64 value
; returns: rax = abs(value)
; clobbers: rax
rational_abs_i64:
	mov	rax, rdi
	test	rax, rax
	jns	.rai_done
	neg	rax
.rai_done:
	ret

; rdi = signed int64 a, rsi = signed int64 b
; returns: rax = gcd(abs(a), abs(b)); 0 only when a == 0 and b == 0
; clobbers: rax, rcx, rdx, rdi, rsi
rational_gcd_i64:
	mov	rcx, rsi
	call	rational_abs_i64
	mov	rdi, rax
	xchg	rdi, rcx
	call	rational_abs_i64
	mov	rsi, rax
	mov	rdi, rcx
.gcd_loop:
	test	rsi, rsi
	jz	.gcd_done
	mov	rax, rdi
	xor	rdx, rdx
	div	rsi
	mov	rdi, rsi
	mov	rsi, rdx
	jmp	.gcd_loop
.gcd_done:
	mov	rax, rdi
	ret

; rdi = Rational*
; returns: rax = RATIONAL_OK or RATIONAL_ERR_ZERO_DEN
; clobbers: rax, rcx, rdx, rsi, rdi
rational_normalize:
	push	rbx
	cmp	qword [rdi + RATIONAL_DEN_OFF], 0
	je	.rn_zero_den
	cmp	qword [rdi + RATIONAL_NUM_OFF], 0
	jne	.rn_sign
	mov	qword [rdi + RATIONAL_DEN_OFF], 1
	xor	rax, rax
	pop	rbx
	ret
.rn_sign:
	cmp	qword [rdi + RATIONAL_DEN_OFF], 0
	jg	.rn_gcd
	neg	qword [rdi + RATIONAL_NUM_OFF]
	neg	qword [rdi + RATIONAL_DEN_OFF]
.rn_gcd:
	mov	rbx, rdi
	mov	rdi, [rbx + RATIONAL_NUM_OFF]
	mov	rsi, [rbx + RATIONAL_DEN_OFF]
	call	rational_gcd_i64
	mov	rsi, rax
	mov	rax, [rbx + RATIONAL_NUM_OFF]
	cqo
	idiv	rsi
	mov	[rbx + RATIONAL_NUM_OFF], rax
	mov	rax, [rbx + RATIONAL_DEN_OFF]
	cqo
	idiv	rsi
	mov	[rbx + RATIONAL_DEN_OFF], rax
	xor	rax, rax
	pop	rbx
	ret
.rn_zero_den:
	mov	rax, RATIONAL_ERR_ZERO_DEN
	pop	rbx
	ret

; rdi = Rational*, rsi = numerator, rdx = denominator
; returns: rax = RATIONAL_OK or RATIONAL_ERR_ZERO_DEN
; clobbers: rax, rcx, rdx, rsi, rdi
rational_set:
	mov	[rdi + RATIONAL_NUM_OFF], rsi
	mov	[rdi + RATIONAL_DEN_OFF], rdx
	jmp	rational_normalize

; rdi = Rational*, rsi = null-terminated string: int64 or int64/int64
; returns: rax = RATIONAL_OK or RATIONAL_ERR_*
; clobbers: rax, rcx, rdx, rsi, rdi, r8-r11
rational_parse_cstr:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	rbx, rdi
	mov	r12, rsi
	mov	r14, 1
	xor	r13, r13
	xor	r15, r15
	movzx	ecx, byte [r12]
	cmp	cl, '-'
	jne	.rpc_num_plus
	mov	r14, -1
	inc	r12
	jmp	.rpc_num_loop
.rpc_num_plus:
	cmp	cl, '+'
	jne	.rpc_num_loop
	inc	r12
.rpc_num_loop:
	movzx	ecx, byte [r12]
	cmp	cl, '0'
	jb	.rpc_num_end
	cmp	cl, '9'
	ja	.rpc_num_end
	imul	r13, r13, 10
	sub	cl, '0'
	add	r13, rcx
	mov	r15, 1
	inc	r12
	jmp	.rpc_num_loop
.rpc_num_end:
	test	r15, r15
	jz	.rpc_parse_err
	imul	r13, r14
	cmp	byte [r12], 0
	je	.rpc_int
	cmp	byte [r12], '/'
	jne	.rpc_parse_err
	inc	r12
	mov	r14, 1
	xor	r8, r8
	xor	r15, r15
	movzx	ecx, byte [r12]
	cmp	cl, '-'
	jne	.rpc_den_plus
	mov	r14, -1
	inc	r12
	jmp	.rpc_den_loop
.rpc_den_plus:
	cmp	cl, '+'
	jne	.rpc_den_loop
	inc	r12
.rpc_den_loop:
	movzx	ecx, byte [r12]
	cmp	cl, '0'
	jb	.rpc_den_end
	cmp	cl, '9'
	ja	.rpc_den_end
	imul	r8, r8, 10
	sub	cl, '0'
	add	r8, rcx
	mov	r15, 1
	inc	r12
	jmp	.rpc_den_loop
.rpc_den_end:
	test	r15, r15
	jz	.rpc_parse_err
	cmp	byte [r12], 0
	jne	.rpc_parse_err
	imul	r8, r14
	mov	rdi, rbx
	mov	rsi, r13
	mov	rdx, r8
	call	rational_set
	jmp	.rpc_done
.rpc_int:
	mov	rdi, rbx
	mov	rsi, r13
	mov	rdx, 1
	call	rational_set
	jmp	.rpc_done
.rpc_parse_err:
	mov	rax, RATIONAL_ERR_PARSE
.rpc_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = dst Rational*, rsi = a Rational*, rdx = b Rational*
; returns: rax = RATIONAL_OK or RATIONAL_ERR_ZERO_DEN
; clobbers: rax, rcx, rdx, r8, r9, r10, r11, rsi, rdi
rational_add:
	mov	r8, [rsi + RATIONAL_NUM_OFF]
	imul	r8, [rdx + RATIONAL_DEN_OFF]
	mov	r9, [rdx + RATIONAL_NUM_OFF]
	imul	r9, [rsi + RATIONAL_DEN_OFF]
	add	r8, r9
	mov	r10, [rsi + RATIONAL_DEN_OFF]
	imul	r10, [rdx + RATIONAL_DEN_OFF]
	mov	rsi, r8
	mov	rdx, r10
	jmp	rational_set

; rdi = dst Rational*, rsi = a Rational*, rdx = b Rational*
; returns: rax = RATIONAL_OK or RATIONAL_ERR_ZERO_DEN
; clobbers: rax, rcx, rdx, r8, r9, r10, r11, rsi, rdi
rational_sub:
	mov	r8, [rsi + RATIONAL_NUM_OFF]
	imul	r8, [rdx + RATIONAL_DEN_OFF]
	mov	r9, [rdx + RATIONAL_NUM_OFF]
	imul	r9, [rsi + RATIONAL_DEN_OFF]
	sub	r8, r9
	mov	r10, [rsi + RATIONAL_DEN_OFF]
	imul	r10, [rdx + RATIONAL_DEN_OFF]
	mov	rsi, r8
	mov	rdx, r10
	jmp	rational_set

; rdi = dst Rational*, rsi = a Rational*, rdx = b Rational*
; returns: rax = RATIONAL_OK or RATIONAL_ERR_ZERO_DEN
; clobbers: rax, rcx, rdx, r8, r9, r10, r11, rsi, rdi
rational_mul:
	mov	r8, [rsi + RATIONAL_NUM_OFF]
	imul	r8, [rdx + RATIONAL_NUM_OFF]
	mov	r9, [rsi + RATIONAL_DEN_OFF]
	imul	r9, [rdx + RATIONAL_DEN_OFF]
	mov	rsi, r8
	mov	rdx, r9
	jmp	rational_set

; rdi = dst Rational*, rsi = a Rational*, rdx = b Rational*
; returns: rax = RATIONAL_OK or RATIONAL_ERR_ZERO_DEN
; clobbers: rax, rcx, rdx, r8, r9, r10, r11, rsi, rdi
rational_div:
	mov	r8, [rsi + RATIONAL_NUM_OFF]
	imul	r8, [rdx + RATIONAL_DEN_OFF]
	mov	r9, [rsi + RATIONAL_DEN_OFF]
	imul	r9, [rdx + RATIONAL_NUM_OFF]
	mov	rsi, r8
	mov	rdx, r9
	jmp	rational_set

; rdi = Rational*
; returns: writes num/den to stdout
; clobbers: delegated to print_int64 and print_char
rational_print:
	push	rbx
	mov	rbx, rdi
	mov	rax, [rdi + RATIONAL_NUM_OFF]
	call	print_int64
	mov	al, '/'
	call	print_char
	mov	rax, [rbx + RATIONAL_DEN_OFF]
	call	print_int64
	pop	rbx
	ret

; rdi = Rational*
; returns: writes num/den and newline to stdout
; clobbers: delegated to rational_print and print_char
rational_print_nl:
	call	rational_print
	mov	al, 10
	jmp	print_char
