; Polynomial arithmetic over Rational coefficients.
;
; Coefficients are stored low-to-high: c0 + c1*x + c2*x^2.
; Storage is caller-owned: Polynomial points at a Rational buffer.
; Destination polynomials must not alias source polynomials.

struc Polynomial
{
	.coeffs dq ?
	.count dq ?
	.cap dq ?
}

virtual at 0
	_Polynomial Polynomial
	POLY_SIZE = $
end virtual

POLY_COEFFS_OFF equ 0
POLY_COUNT_OFF equ 8
POLY_CAP_OFF equ 16

POLY_OK equ 0
POLY_ERR_CAP equ -1
POLY_ERR_RATIONAL equ -2

; rdi = Polynomial*, rsi = coefficient index
; returns: rax = Rational*
; clobbers: rax, rcx
poly_coeff_ptr:
	mov	rax, [rdi + POLY_COEFFS_OFF]
	mov	rcx, rsi
	imul	rcx, rcx, RATIONAL_SIZE
	add	rax, rcx
	ret

; rdi = Polynomial*, rsi = Rational* coeff buffer, rdx = capacity
; returns: rax = POLY_OK or POLY_ERR_CAP/RATIONAL
; clobbers: rax, rcx, rdx, rsi, rdi
poly_init:
	push	rbx
	mov	rbx, rdi
	test	rdx, rdx
	jz	.pi_cap_err
	mov	[rbx + POLY_COEFFS_OFF], rsi
	mov	[rbx + POLY_CAP_OFF], rdx
	mov	qword [rbx + POLY_COUNT_OFF], 1
	mov	rdi, rsi
	xor	rsi, rsi
	mov	rdx, 1
	call	rational_set
	test	rax, rax
	jnz	.pi_rat_err
	xor	rax, rax
	pop	rbx
	ret
.pi_cap_err:
	mov	rax, POLY_ERR_CAP
	pop	rbx
	ret
.pi_rat_err:
	mov	rax, POLY_ERR_RATIONAL
	pop	rbx
	ret

; rdi = Polynomial*, rsi = count
; returns: rax = POLY_OK or POLY_ERR_RATIONAL
; clobbers: rax, rcx, rdx, rsi, rdi
poly_clear_n:
	push	rbx
	push	r12
	push	r13
	mov	rbx, rdi
	mov	r12, rsi
	xor	r13, r13
.pcn_loop:
	cmp	r13, r12
	jae	.pcn_done
	mov	rdi, rbx
	mov	rsi, r13
	call	poly_coeff_ptr
	mov	rdi, rax
	xor	rsi, rsi
	mov	rdx, 1
	call	rational_set
	test	rax, rax
	jnz	.pcn_rat_err
	inc	r13
	jmp	.pcn_loop
.pcn_done:
	xor	rax, rax
	pop	r13
	pop	r12
	pop	rbx
	ret
.pcn_rat_err:
	mov	rax, POLY_ERR_RATIONAL
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = Polynomial*
; returns: rax = POLY_OK
; clobbers: rax, rcx, rdx, rsi, rdi
poly_normalize:
	mov	rcx, [rdi + POLY_COUNT_OFF]
	cmp	rcx, 1
	jbe	.pn_done
.pn_loop:
	cmp	rcx, 1
	jbe	.pn_store
	mov	rax, rcx
	dec	rax
	mov	rdx, [rdi + POLY_COEFFS_OFF]
	imul	rax, rax, RATIONAL_SIZE
	add	rdx, rax
	cmp	qword [rdx + RATIONAL_NUM_OFF], 0
	jne	.pn_store
	dec	rcx
	jmp	.pn_loop
.pn_store:
	mov	[rdi + POLY_COUNT_OFF], rcx
.pn_done:
	xor	rax, rax
	ret

; rdi = Polynomial*, rsi = degree, rdx = numerator, rcx = denominator
; returns: rax = POLY_OK or POLY_ERR_CAP/RATIONAL
; clobbers: rax, rcx, rdx, rsi, rdi, r8-r11
poly_set_coeff_i64:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, rdx
	mov	r14, rcx
	cmp	r12, [rbx + POLY_CAP_OFF]
	jae	.psc_cap_err
	mov	r15, [rbx + POLY_COUNT_OFF]
	cmp	r12, r15
	jb	.psc_set
.psc_fill:
	cmp	r15, r12
	ja	.psc_count_ready
	mov	rdi, rbx
	mov	rsi, r15
	call	poly_coeff_ptr
	mov	rdi, rax
	xor	rsi, rsi
	mov	rdx, 1
	call	rational_set
	test	rax, rax
	jnz	.psc_rat_err
	inc	r15
	jmp	.psc_fill
.psc_count_ready:
	lea	rax, [r12 + 1]
	mov	[rbx + POLY_COUNT_OFF], rax
.psc_set:
	mov	rdi, rbx
	mov	rsi, r12
	call	poly_coeff_ptr
	mov	rdi, rax
	mov	rsi, r13
	mov	rdx, r14
	call	rational_set
	test	rax, rax
	jnz	.psc_rat_err
	mov	rdi, rbx
	call	poly_normalize
	xor	rax, rax
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
.psc_cap_err:
	mov	rax, POLY_ERR_CAP
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
.psc_rat_err:
	mov	rax, POLY_ERR_RATIONAL
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = dst Polynomial*, rsi = a Polynomial*, rdx = b Polynomial*
; returns: rax = POLY_OK or POLY_ERR_CAP/RATIONAL
; clobbers: rax, rcx, rdx, rsi, rdi, r8-r11
poly_add:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, rdx
	mov	r14, [r12 + POLY_COUNT_OFF]
	mov	rax, [r13 + POLY_COUNT_OFF]
	cmp	r14, rax
	jae	.pa_have_count
	mov	r14, rax
.pa_have_count:
	cmp	r14, [rbx + POLY_CAP_OFF]
	ja	.pa_cap_err
	mov	rdi, rbx
	mov	rsi, r14
	call	poly_clear_n
	test	rax, rax
	jnz	.pa_done
	xor	r15, r15
.pa_loop:
	cmp	r15, r14
	jae	.pa_finish
	mov	rdi, rbx
	mov	rsi, r15
	call	poly_coeff_ptr
	mov	r8, rax
	mov	rdi, r12
	mov	rsi, r15
	call	poly_coeff_ptr
	mov	r9, rax
	mov	rdi, r13
	mov	rsi, r15
	call	poly_coeff_ptr
	mov	r10, rax
	cmp	r15, [r12 + POLY_COUNT_OFF]
	jae	.pa_copy_b
	cmp	r15, [r13 + POLY_COUNT_OFF]
	jae	.pa_copy_a
	mov	rdi, r8
	mov	rsi, r9
	mov	rdx, r10
	call	rational_add
	test	rax, rax
	jnz	.pa_rat_err
	jmp	.pa_next
.pa_copy_a:
	mov	rdi, r8
	mov	rsi, [r9 + RATIONAL_NUM_OFF]
	mov	rdx, [r9 + RATIONAL_DEN_OFF]
	call	rational_set
	test	rax, rax
	jnz	.pa_rat_err
	jmp	.pa_next
.pa_copy_b:
	cmp	r15, [r13 + POLY_COUNT_OFF]
	jae	.pa_next
	mov	rdi, r8
	mov	rsi, [r10 + RATIONAL_NUM_OFF]
	mov	rdx, [r10 + RATIONAL_DEN_OFF]
	call	rational_set
	test	rax, rax
	jnz	.pa_rat_err
.pa_next:
	inc	r15
	jmp	.pa_loop
.pa_finish:
	mov	[rbx + POLY_COUNT_OFF], r14
	mov	rdi, rbx
	call	poly_normalize
	xor	rax, rax
	jmp	.pa_done
.pa_cap_err:
	mov	rax, POLY_ERR_CAP
	jmp	.pa_done
.pa_rat_err:
	mov	rax, POLY_ERR_RATIONAL
.pa_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = dst Polynomial*, rsi = a Polynomial*, rdx = b Polynomial*
; returns: rax = POLY_OK or POLY_ERR_CAP/RATIONAL
; clobbers: rax, rcx, rdx, rsi, rdi, r8-r11
poly_mul:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, RATIONAL_SIZE + 8
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, rdx
	mov	r14, [r12 + POLY_COUNT_OFF]
	add	r14, [r13 + POLY_COUNT_OFF]
	dec	r14
	cmp	r14, [rbx + POLY_CAP_OFF]
	ja	.pm_cap_err
	mov	rdi, rbx
	mov	rsi, r14
	call	poly_clear_n
	test	rax, rax
	jnz	.pm_done
	xor	r15, r15
.pm_outer:
	cmp	r15, [r12 + POLY_COUNT_OFF]
	jae	.pm_finish
	mov	qword [rsp + RATIONAL_SIZE], 0
.pm_inner:
	mov	r11, [rsp + RATIONAL_SIZE]
	cmp	r11, [r13 + POLY_COUNT_OFF]
	jae	.pm_next_outer
	mov	rdi, r12
	mov	rsi, r15
	call	poly_coeff_ptr
	mov	r8, rax
	mov	rdi, r13
	mov	rsi, r11
	call	poly_coeff_ptr
	mov	r9, rax
	mov	rdi, rsp
	mov	rsi, r8
	mov	rdx, r9
	call	rational_mul
	test	rax, rax
	jnz	.pm_rat_err
	mov	r10, r15
	add	r10, r11
	mov	rdi, rbx
	mov	rsi, r10
	call	poly_coeff_ptr
	mov	rdi, rax
	mov	rsi, rax
	mov	rdx, rsp
	call	rational_add
	test	rax, rax
	jnz	.pm_rat_err
	inc	qword [rsp + RATIONAL_SIZE]
	jmp	.pm_inner
.pm_next_outer:
	inc	r15
	jmp	.pm_outer
.pm_finish:
	mov	[rbx + POLY_COUNT_OFF], r14
	mov	rdi, rbx
	call	poly_normalize
	xor	rax, rax
	jmp	.pm_done
.pm_cap_err:
	mov	rax, POLY_ERR_CAP
	jmp	.pm_done
.pm_rat_err:
	mov	rax, POLY_ERR_RATIONAL
.pm_done:
	add	rsp, RATIONAL_SIZE + 8
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = dst Polynomial*, rsi = source Polynomial*
; returns: rax = POLY_OK or POLY_ERR_CAP/RATIONAL
; clobbers: rax, rcx, rdx, rsi, rdi, r8-r11
poly_derivative:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, [r12 + POLY_COUNT_OFF]
	cmp	r13, 1
	ja	.pd_nonzero
	cmp	qword [rbx + POLY_CAP_OFF], 1
	jb	.pd_cap_err
	mov	rdi, rbx
	mov	rsi, 1
	call	poly_clear_n
	test	rax, rax
	jnz	.pd_done
	mov	qword [rbx + POLY_COUNT_OFF], 1
	xor	rax, rax
	jmp	.pd_done
.pd_nonzero:
	mov	r14, r13
	dec	r14
	cmp	r14, [rbx + POLY_CAP_OFF]
	ja	.pd_cap_err
	mov	rdi, rbx
	mov	rsi, r14
	call	poly_clear_n
	test	rax, rax
	jnz	.pd_done
	mov	r15, 1
.pd_loop:
	cmp	r15, r13
	jae	.pd_finish
	mov	rdi, r12
	mov	rsi, r15
	call	poly_coeff_ptr
	mov	r8, rax
	mov	rdi, rbx
	mov	rsi, r15
	dec	rsi
	call	poly_coeff_ptr
	mov	rdi, rax
	mov	rsi, [r8 + RATIONAL_NUM_OFF]
	imul	rsi, r15
	mov	rdx, [r8 + RATIONAL_DEN_OFF]
	call	rational_set
	test	rax, rax
	jnz	.pd_rat_err
	inc	r15
	jmp	.pd_loop
.pd_finish:
	mov	[rbx + POLY_COUNT_OFF], r14
	mov	rdi, rbx
	call	poly_normalize
	xor	rax, rax
	jmp	.pd_done
.pd_cap_err:
	mov	rax, POLY_ERR_CAP
	jmp	.pd_done
.pd_rat_err:
	mov	rax, POLY_ERR_RATIONAL
.pd_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = dst Polynomial*, rsi = source Polynomial*
; returns: rax = POLY_OK or POLY_ERR_CAP/RATIONAL
; clobbers: rax, rcx, rdx, rsi, rdi, r8-r11
poly_integral:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, [r12 + POLY_COUNT_OFF]
	lea	rax, [r13 + 1]
	cmp	rax, [rbx + POLY_CAP_OFF]
	ja	.pin_cap_err
	mov	rdi, rbx
	mov	rsi, rax
	call	poly_clear_n
	test	rax, rax
	jnz	.pin_done
	xor	r14, r14
.pin_loop:
	cmp	r14, r13
	jae	.pin_finish
	mov	rdi, r12
	mov	rsi, r14
	call	poly_coeff_ptr
	mov	r8, rax
	mov	rdi, rbx
	mov	rsi, r14
	inc	rsi
	call	poly_coeff_ptr
	mov	rdi, rax
	mov	rsi, [r8 + RATIONAL_NUM_OFF]
	mov	rdx, [r8 + RATIONAL_DEN_OFF]
	lea	r10, [r14 + 1]
	imul	rdx, r10
	call	rational_set
	test	rax, rax
	jnz	.pin_rat_err
	inc	r14
	jmp	.pin_loop
.pin_finish:
	lea	rax, [r13 + 1]
	mov	[rbx + POLY_COUNT_OFF], rax
	mov	rdi, rbx
	call	poly_normalize
	xor	rax, rax
	jmp	.pin_done
.pin_cap_err:
	mov	rax, POLY_ERR_CAP
	jmp	.pin_done
.pin_rat_err:
	mov	rax, POLY_ERR_RATIONAL
.pin_done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = dst Rational*, rsi = Polynomial*, rdx = x Rational*
; returns: rax = POLY_OK or POLY_ERR_RATIONAL
; clobbers: rax, rcx, rdx, rsi, rdi, r8-r11
poly_eval:
	push	rbx
	push	r12
	push	r13
	push	r14
	sub	rsp, RATIONAL_SIZE
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, rdx
	mov	rdi, rbx
	xor	rsi, rsi
	mov	rdx, 1
	call	rational_set
	test	rax, rax
	jnz	.pe_rat_err
	mov	r14, [r12 + POLY_COUNT_OFF]
.pe_loop:
	test	r14, r14
	jz	.pe_done_ok
	dec	r14
	mov	rdi, rsp
	mov	rsi, rbx
	mov	rdx, r13
	call	rational_mul
	test	rax, rax
	jnz	.pe_rat_err
	mov	rdi, r12
	mov	rsi, r14
	call	poly_coeff_ptr
	mov	rdi, rbx
	mov	rsi, rsp
	mov	rdx, rax
	call	rational_add
	test	rax, rax
	jnz	.pe_rat_err
	jmp	.pe_loop
.pe_done_ok:
	xor	rax, rax
	jmp	.pe_done
.pe_rat_err:
	mov	rax, POLY_ERR_RATIONAL
.pe_done:
	add	rsp, RATIONAL_SIZE
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = dst Rational*, rsi = Polynomial*, rdx = a Rational*, rcx = b Rational*
; returns: rax = POLY_OK or POLY_ERR_RATIONAL
; clobbers: rax, rcx, rdx, rsi, rdi, r8-r11
poly_definite_integral:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, RATIONAL_SIZE * 6
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, rdx
	mov	r14, rcx
	; stack slots: 0=sum, 16=powa, 32=powb, 48=coefdiv, 64=diff, 80=term
	lea	rdi, [rsp]
	xor	rsi, rsi
	mov	rdx, 1
	call	rational_set
	test	rax, rax
	jnz	.pdi_rat_err
	lea	rdi, [rsp + 16]
	mov	rsi, 1
	mov	rdx, 1
	call	rational_set
	test	rax, rax
	jnz	.pdi_rat_err
	lea	rdi, [rsp + 32]
	mov	rsi, 1
	mov	rdx, 1
	call	rational_set
	test	rax, rax
	jnz	.pdi_rat_err
	xor	r15, r15
.pdi_loop:
	cmp	r15, [r12 + POLY_COUNT_OFF]
	jae	.pdi_copy_result
	lea	rdi, [rsp + 16]
	lea	rsi, [rsp + 16]
	mov	rdx, r13
	call	rational_mul
	test	rax, rax
	jnz	.pdi_rat_err
	lea	rdi, [rsp + 32]
	lea	rsi, [rsp + 32]
	mov	rdx, r14
	call	rational_mul
	test	rax, rax
	jnz	.pdi_rat_err
	mov	rdi, r12
	mov	rsi, r15
	call	poly_coeff_ptr
	mov	r8, rax
	lea	rdi, [rsp + 48]
	mov	rsi, [r8 + RATIONAL_NUM_OFF]
	mov	rdx, [r8 + RATIONAL_DEN_OFF]
	lea	r10, [r15 + 1]
	imul	rdx, r10
	call	rational_set
	test	rax, rax
	jnz	.pdi_rat_err
	lea	rdi, [rsp + 64]
	lea	rsi, [rsp + 32]
	lea	rdx, [rsp + 16]
	call	rational_sub
	test	rax, rax
	jnz	.pdi_rat_err
	lea	rdi, [rsp + 80]
	lea	rsi, [rsp + 48]
	lea	rdx, [rsp + 64]
	call	rational_mul
	test	rax, rax
	jnz	.pdi_rat_err
	lea	rdi, [rsp]
	lea	rsi, [rsp]
	lea	rdx, [rsp + 80]
	call	rational_add
	test	rax, rax
	jnz	.pdi_rat_err
	inc	r15
	jmp	.pdi_loop
.pdi_copy_result:
	mov	rdi, rbx
	mov	rsi, [rsp + RATIONAL_NUM_OFF]
	mov	rdx, [rsp + RATIONAL_DEN_OFF]
	call	rational_set
	test	rax, rax
	jnz	.pdi_rat_err
	xor	rax, rax
	jmp	.pdi_done
.pdi_rat_err:
	mov	rax, POLY_ERR_RATIONAL
.pdi_done:
	add	rsp, RATIONAL_SIZE * 6
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = Rational*
; returns: writes abs(coeff), omitting /1
; clobbers: delegated to print_int64 and print_char
poly_print_abs_rational:
	push	rbx
	mov	rbx, rdi
	mov	rax, [rbx + RATIONAL_NUM_OFF]
	test	rax, rax
	jns	.ppar_num
	neg	rax
.ppar_num:
	call	print_int64
	cmp	qword [rbx + RATIONAL_DEN_OFF], 1
	je	.ppar_done
	mov	al, '/'
	call	print_char
	mov	rax, [rbx + RATIONAL_DEN_OFF]
	call	print_int64
.ppar_done:
	pop	rbx
	ret

; rdi = Polynomial*
; returns: writes compact polynomial to stdout
; clobbers: delegated to print helpers
poly_print:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, [r12 + POLY_COUNT_OFF]
	mov	r14, 1
.pp_loop:
	test	r13, r13
	jz	.pp_done
	dec	r13
	mov	rdi, r12
	mov	rsi, r13
	call	poly_coeff_ptr
	mov	r15, rax
	mov	rax, [r15 + RATIONAL_NUM_OFF]
	test	rax, rax
	jz	.pp_loop
	test	r14, r14
	jz	.pp_sign_mid
	test	rax, rax
	jns	.pp_term
	mov	al, '-'
	call	print_char
	jmp	.pp_term
.pp_sign_mid:
	test	rax, rax
	jns	.pp_plus
	mov	al, '-'
	call	print_char
	jmp	.pp_term
.pp_plus:
	mov	al, '+'
	call	print_char
.pp_term:
	xor	r14, r14
	mov	rax, [r15 + RATIONAL_NUM_OFF]
	test	rax, rax
	jns	.pp_abs_ready
	neg	rax
.pp_abs_ready:
	cmp	r13, 0
	je	.pp_print_coeff
	cmp	rax, 1
	jne	.pp_print_coeff_var
	cmp	qword [r15 + RATIONAL_DEN_OFF], 1
	je	.pp_var
.pp_print_coeff_var:
	mov	rdi, r15
	call	poly_print_abs_rational
	jmp	.pp_var
.pp_print_coeff:
	mov	rdi, r15
	call	poly_print_abs_rational
	jmp	.pp_loop
.pp_var:
	mov	al, 'x'
	call	print_char
	cmp	r13, 1
	jbe	.pp_loop
	mov	al, '^'
	call	print_char
	mov	rax, r13
	call	print_int64
	jmp	.pp_loop
.pp_done:
	test	r14, r14
	jz	.pp_ret
	mov	al, '0'
	call	print_char
.pp_ret:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = Polynomial*
; returns: writes polynomial and newline to stdout
; clobbers: delegated to poly_print and print_char
poly_print_nl:
	call	poly_print
	mov	al, 10
	jmp	print_char
