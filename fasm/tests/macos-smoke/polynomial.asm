; Smoke: exact polynomial arithmetic over Rational coefficients.

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
	lea	rdi, [poly_q]
	lea	rsi, [buf_q]
	mov	rdx, 8
	call	poly_init
	lea	rdi, [poly_r]
	lea	rsi, [buf_r]
	mov	rdx, 16
	call	poly_init
	lea	rdi, [poly_s]
	lea	rsi, [buf_s]
	mov	rdx, 16
	call	poly_init

	; p(x) = 2x^2 + 3x + 1
	lea	rdi, [poly_p]
	mov	rsi, 0
	mov	rdx, 1
	mov	rcx, 1
	call	poly_set_coeff_i64
	lea	rdi, [poly_p]
	mov	rsi, 1
	mov	rdx, 3
	mov	rcx, 1
	call	poly_set_coeff_i64
	lea	rdi, [poly_p]
	mov	rsi, 2
	mov	rdx, 2
	mov	rcx, 1
	call	poly_set_coeff_i64
	lea	rdi, [msg_p]
	call	print_cstr
	lea	rdi, [poly_p]
	call	poly_print_nl

	lea	rdi, [poly_r]
	lea	rsi, [poly_p]
	call	poly_derivative
	lea	rdi, [msg_derivative]
	call	print_cstr
	lea	rdi, [poly_r]
	call	poly_print_nl

	lea	rdi, [poly_r]
	lea	rsi, [poly_p]
	call	poly_integral
	lea	rdi, [msg_integral]
	call	print_cstr
	lea	rdi, [poly_r]
	call	poly_print_nl

	lea	rdi, [x_value]
	mov	rsi, 2
	mov	rdx, 1
	call	rational_set
	lea	rdi, [eval_result]
	lea	rsi, [poly_p]
	lea	rdx, [x_value]
	call	poly_eval
	lea	rdi, [msg_eval]
	call	print_cstr
	lea	rdi, [eval_result]
	call	rational_print_nl

	lea	rdi, [a_value]
	mov	rsi, 0
	mov	rdx, 1
	call	rational_set
	lea	rdi, [b_value]
	mov	rsi, 1
	mov	rdx, 1
	call	rational_set
	lea	rdi, [eval_result]
	lea	rsi, [poly_p]
	lea	rdx, [a_value]
	lea	rcx, [b_value]
	call	poly_definite_integral
	lea	rdi, [msg_definite]
	call	print_cstr
	lea	rdi, [eval_result]
	call	rational_print_nl

	; q(x) = x + 1
	lea	rdi, [poly_q]
	mov	rsi, 0
	mov	rdx, 1
	mov	rcx, 1
	call	poly_set_coeff_i64
	lea	rdi, [poly_q]
	mov	rsi, 1
	mov	rdx, 1
	mov	rcx, 1
	call	poly_set_coeff_i64

	lea	rdi, [poly_r]
	lea	rsi, [poly_p]
	lea	rdx, [poly_q]
	call	poly_add
	lea	rdi, [msg_add]
	call	print_cstr
	lea	rdi, [poly_r]
	call	poly_print_nl

	lea	rdi, [poly_s]
	lea	rsi, [poly_p]
	lea	rdx, [poly_q]
	call	poly_mul
	lea	rdi, [msg_mul]
	call	print_cstr
	lea	rdi, [poly_s]
	call	poly_print_nl

	exit EXIT_SUCCESS

segment readable writeable

msg_p db 'p = ', 0
msg_derivative db 'p', 39, ' = ', 0
msg_integral db 'integral p dx = ', 0
msg_eval db 'p(2) = ', 0
msg_definite db 'integral 0..1 p dx = ', 0
msg_add db 'p + (x+1) = ', 0
msg_mul db 'p * (x+1) = ', 0

poly_p rb POLY_SIZE
poly_q rb POLY_SIZE
poly_r rb POLY_SIZE
poly_s rb POLY_SIZE

buf_p rb RATIONAL_SIZE * 8
buf_q rb RATIONAL_SIZE * 8
buf_r rb RATIONAL_SIZE * 16
buf_s rb RATIONAL_SIZE * 16

x_value rb RATIONAL_SIZE
a_value rb RATIONAL_SIZE
b_value rb RATIONAL_SIZE
eval_result rb RATIONAL_SIZE

include "fasm/core/runtime_bss.inc"
runtime_print_bss
