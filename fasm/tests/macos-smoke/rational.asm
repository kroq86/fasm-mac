; Smoke: exact rational arithmetic and simple polynomial integrals.

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/rational.inc"

entry start

start:
	lea	rdi, [frac_a]
	mov	rsi, -6
	mov	rdx, 8
	call	rational_set
	lea	rdi, [msg_reduce]
	call	print_cstr
	lea	rdi, [frac_a]
	call	rational_print_nl

	lea	rdi, [frac_a]
	mov	rsi, 1
	mov	rdx, 3
	call	rational_set
	lea	rdi, [frac_b]
	mov	rsi, 1
	mov	rdx, 6
	call	rational_set
	lea	rdi, [frac_result]
	lea	rsi, [frac_a]
	lea	rdx, [frac_b]
	call	rational_add
	lea	rdi, [msg_add]
	call	print_cstr
	lea	rdi, [frac_result]
	call	rational_print_nl

	; integral_0^1 x dx = 1/2
	lea	rdi, [frac_result]
	mov	rsi, 1
	mov	rdx, 2
	call	rational_set
	lea	rdi, [msg_integral_x]
	call	print_cstr
	lea	rdi, [frac_result]
	call	rational_print_nl

	; integral_0^1 (2x^2 + 3x + 1) dx = 2/3 + 3/2 + 1 = 19/6
	lea	rdi, [poly_sum]
	mov	rsi, 0
	mov	rdx, 1
	call	rational_set
	lea	rdi, [term]
	mov	rsi, 2
	mov	rdx, 3
	call	rational_set
	lea	rdi, [poly_sum]
	lea	rsi, [poly_sum]
	lea	rdx, [term]
	call	rational_add
	lea	rdi, [term]
	mov	rsi, 3
	mov	rdx, 2
	call	rational_set
	lea	rdi, [poly_sum]
	lea	rsi, [poly_sum]
	lea	rdx, [term]
	call	rational_add
	lea	rdi, [term]
	mov	rsi, 1
	mov	rdx, 1
	call	rational_set
	lea	rdi, [poly_sum]
	lea	rsi, [poly_sum]
	lea	rdx, [term]
	call	rational_add
	lea	rdi, [msg_integral_poly]
	call	print_cstr
	lea	rdi, [poly_sum]
	call	rational_print_nl

	exit EXIT_SUCCESS

segment readable writeable

msg_reduce db '-6/8 -> ', 0
msg_add db '1/3 + 1/6 = ', 0
msg_integral_x db 'integral 0..1 x dx = ', 0
msg_integral_poly db 'integral 0..1 (2x^2 + 3x + 1) dx = ', 0

frac_a rb RATIONAL_SIZE
frac_b rb RATIONAL_SIZE
frac_result rb RATIONAL_SIZE
poly_sum rb RATIONAL_SIZE
term rb RATIONAL_SIZE

include "fasm/core/runtime_bss.inc"
runtime_print_bss
