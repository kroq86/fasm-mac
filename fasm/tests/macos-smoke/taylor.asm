; Smoke: Taylor coefficients of a polynomial at a rational point.
;
; p(x) = x^3 - 2x + 1.
; Around a = 1, with h = x - 1:
; p(1+h) = h^3 + 3h^2 + h.

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/rational.inc"
include "fasm/core/polynomial.inc"

entry start

start:
	lea	rdi, [poly_p]
	lea	rsi, [buf_p]
	mov	rdx, 8
	call	poly_init
	lea	rdi, [poly_d1]
	lea	rsi, [buf_d1]
	mov	rdx, 8
	call	poly_init
	lea	rdi, [poly_d2]
	lea	rsi, [buf_d2]
	mov	rdx, 8
	call	poly_init
	lea	rdi, [poly_d3]
	lea	rsi, [buf_d3]
	mov	rdx, 8
	call	poly_init
	lea	rdi, [poly_taylor]
	lea	rsi, [buf_taylor]
	mov	rdx, 8
	call	poly_init

	; p(x) = x^3 - 2x + 1
	lea	rdi, [poly_p]
	mov	rsi, 0
	mov	rdx, 1
	mov	rcx, 1
	call	poly_set_coeff_i64
	lea	rdi, [poly_p]
	mov	rsi, 1
	mov	rdx, -2
	mov	rcx, 1
	call	poly_set_coeff_i64
	lea	rdi, [poly_p]
	mov	rsi, 3
	mov	rdx, 1
	mov	rcx, 1
	call	poly_set_coeff_i64

	lea	rdi, [point_a]
	mov	rsi, 1
	mov	rdx, 1
	call	rational_set

	; c0 = p(a)
	lea	rdi, [eval_result]
	lea	rsi, [poly_p]
	lea	rdx, [point_a]
	call	poly_eval
	lea	rdi, [poly_taylor]
	mov	rsi, 0
	mov	rdx, qword [eval_result + RATIONAL_NUM_OFF]
	mov	rcx, qword [eval_result + RATIONAL_DEN_OFF]
	call	poly_set_coeff_i64

	lea	rdi, [poly_d1]
	lea	rsi, [poly_p]
	call	poly_derivative

	; c1 = p'(a)
	lea	rdi, [eval_result]
	lea	rsi, [poly_d1]
	lea	rdx, [point_a]
	call	poly_eval
	lea	rdi, [poly_taylor]
	mov	rsi, 1
	mov	rdx, qword [eval_result + RATIONAL_NUM_OFF]
	mov	rcx, qword [eval_result + RATIONAL_DEN_OFF]
	call	poly_set_coeff_i64

	lea	rdi, [poly_d2]
	lea	rsi, [poly_d1]
	call	poly_derivative

	; c2 = p''(a) / 2!
	lea	rdi, [eval_result]
	lea	rsi, [poly_d2]
	lea	rdx, [point_a]
	call	poly_eval
	lea	rdi, [poly_taylor]
	mov	rsi, 2
	mov	rdx, qword [eval_result + RATIONAL_NUM_OFF]
	mov	rcx, qword [eval_result + RATIONAL_DEN_OFF]
	imul	rcx, 2
	call	poly_set_coeff_i64

	lea	rdi, [poly_d3]
	lea	rsi, [poly_d2]
	call	poly_derivative

	; c3 = p'''(a) / 3!
	lea	rdi, [eval_result]
	lea	rsi, [poly_d3]
	lea	rdx, [point_a]
	call	poly_eval
	lea	rdi, [poly_taylor]
	mov	rsi, 3
	mov	rdx, qword [eval_result + RATIONAL_NUM_OFF]
	mov	rcx, qword [eval_result + RATIONAL_DEN_OFF]
	imul	rcx, 6
	call	poly_set_coeff_i64

	lea	rdi, [msg_p]
	call	print_cstr
	lea	rdi, [poly_p]
	call	poly_print_nl
	lea	rdi, [msg_taylor]
	call	print_cstr
	lea	rdi, [poly_taylor]
	call	poly_print_nl

	exit EXIT_SUCCESS

segment readable writeable

msg_p db 'demidovich-taylor p = ', 0
msg_taylor db 'taylor at 1 in h = ', 0

poly_p rb POLY_SIZE
poly_d1 rb POLY_SIZE
poly_d2 rb POLY_SIZE
poly_d3 rb POLY_SIZE
poly_taylor rb POLY_SIZE

buf_p rb RATIONAL_SIZE * 8
buf_d1 rb RATIONAL_SIZE * 8
buf_d2 rb RATIONAL_SIZE * 8
buf_d3 rb RATIONAL_SIZE * 8
buf_taylor rb RATIONAL_SIZE * 8

point_a rb RATIONAL_SIZE
eval_result rb RATIONAL_SIZE

include "fasm/core/runtime_bss.inc"
runtime_print_bss
