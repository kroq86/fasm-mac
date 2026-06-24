; fmath: tiny exact math CLI.
;
; Usage:
;   fmath frac <add|sub|mul|div> <a> <b>
;   fmath poly-derive <c0> <c1> ...
;   fmath poly-integrate <c0> <c1> ...
;   fmath poly-eval <x> <c0> <c1> ...

format ELF64 executable 3
include "fasm/core/platform.inc"

POLY_MAX_COEFFS equ 32
POLY_WORK_CAP equ 64

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/rational.inc"
include "fasm/core/polynomial.inc"

entry start

start:
	mov	[argv_base], rsp
	mov	rbx, rsp
	mov	rax, [rbx]
	mov	[argc], rax
	cmp	rax, 2
	jb	usage
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_frac]
	call	str_eq
	test	rax, rax
	jnz	run_frac
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_poly_derive]
	call	str_eq
	test	rax, rax
	jnz	run_poly_derive
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_poly_integrate]
	call	str_eq
	test	rax, rax
	jnz	run_poly_integrate
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 16]
	lea	rsi, [cmd_poly_eval]
	call	str_eq
	test	rax, rax
	jnz	run_poly_eval
	jmp	usage

run_frac:
	cmp	qword [argc], 5
	jne	usage
	mov	rbx, [argv_base]
	lea	rdi, [rat_a]
	mov	rsi, [rbx + 8 + 3 * 8]
	call	rational_parse_cstr
	test	rax, rax
	jnz	parse_error
	mov	rbx, [argv_base]
	lea	rdi, [rat_b]
	mov	rsi, [rbx + 8 + 4 * 8]
	call	rational_parse_cstr
	test	rax, rax
	jnz	parse_error
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 8 + 2 * 8]
	lea	rsi, [op_add]
	call	str_eq
	test	rax, rax
	jnz	.frac_add
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 8 + 2 * 8]
	lea	rsi, [op_sub]
	call	str_eq
	test	rax, rax
	jnz	.frac_sub
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 8 + 2 * 8]
	lea	rsi, [op_mul]
	call	str_eq
	test	rax, rax
	jnz	.frac_mul
	mov	rbx, [argv_base]
	mov	rdi, [rbx + 8 + 2 * 8]
	lea	rsi, [op_div]
	call	str_eq
	test	rax, rax
	jnz	.frac_div
	jmp	usage
.frac_add:
	lea	rdi, [rat_result]
	lea	rsi, [rat_a]
	lea	rdx, [rat_b]
	call	rational_add
	jmp	.frac_done
.frac_sub:
	lea	rdi, [rat_result]
	lea	rsi, [rat_a]
	lea	rdx, [rat_b]
	call	rational_sub
	jmp	.frac_done
.frac_mul:
	lea	rdi, [rat_result]
	lea	rsi, [rat_a]
	lea	rdx, [rat_b]
	call	rational_mul
	jmp	.frac_done
.frac_div:
	lea	rdi, [rat_result]
	lea	rsi, [rat_a]
	lea	rdx, [rat_b]
	call	rational_div
.frac_done:
	test	rax, rax
	jnz	parse_error
	lea	rdi, [rat_result]
	call	rational_print_nl
	exit EXIT_SUCCESS

run_poly_derive:
	mov	rax, [argc]
	cmp	rax, 3
	jb	usage
	sub	rax, 2
	cmp	rax, POLY_MAX_COEFFS
	ja	too_many_error
	mov	[poly_arg_count], rax
	call	init_polys
	lea	rdi, [poly_in]
	mov	rsi, 2
	mov	rdx, [poly_arg_count]
	call	parse_poly_args
	test	rax, rax
	jnz	parse_error
	lea	rdi, [poly_out]
	lea	rsi, [poly_in]
	call	poly_derivative
	test	rax, rax
	jnz	parse_error
	lea	rdi, [poly_out]
	call	poly_print_nl
	exit EXIT_SUCCESS

run_poly_integrate:
	mov	rax, [argc]
	cmp	rax, 3
	jb	usage
	sub	rax, 2
	cmp	rax, POLY_MAX_COEFFS
	ja	too_many_error
	mov	[poly_arg_count], rax
	call	init_polys
	lea	rdi, [poly_in]
	mov	rsi, 2
	mov	rdx, [poly_arg_count]
	call	parse_poly_args
	test	rax, rax
	jnz	parse_error
	lea	rdi, [poly_out]
	lea	rsi, [poly_in]
	call	poly_integral
	test	rax, rax
	jnz	parse_error
	lea	rdi, [poly_out]
	call	poly_print_nl
	exit EXIT_SUCCESS

run_poly_eval:
	mov	rax, [argc]
	cmp	rax, 4
	jb	usage
	sub	rax, 3
	cmp	rax, POLY_MAX_COEFFS
	ja	too_many_error
	mov	[poly_arg_count], rax
	call	init_polys
	mov	rbx, [argv_base]
	lea	rdi, [rat_x]
	mov	rsi, [rbx + 8 + 2 * 8]
	call	rational_parse_cstr
	test	rax, rax
	jnz	parse_error
	lea	rdi, [poly_in]
	mov	rsi, 3
	mov	rdx, [poly_arg_count]
	call	parse_poly_args
	test	rax, rax
	jnz	parse_error
	lea	rdi, [rat_result]
	lea	rsi, [poly_in]
	lea	rdx, [rat_x]
	call	poly_eval
	test	rax, rax
	jnz	parse_error
	lea	rdi, [rat_result]
	call	rational_print_nl
	exit EXIT_SUCCESS

init_polys:
	lea	rdi, [poly_in]
	lea	rsi, [poly_in_buf]
	mov	rdx, POLY_MAX_COEFFS
	call	poly_init
	test	rax, rax
	jnz	parse_error
	lea	rdi, [poly_out]
	lea	rsi, [poly_out_buf]
	mov	rdx, POLY_WORK_CAP
	call	poly_init
	test	rax, rax
	jnz	parse_error
	ret

; rdi = Polynomial*, rsi = argv start index, rdx = coefficient count
; returns: rax = 0 or error
parse_poly_args:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	xor	r15, r15
.ppa_loop:
	cmp	r15, r14
	jae	.ppa_done
	mov	rbx, [argv_base]
	mov	rax, r13
	add	rax, r15
	lea	rdi, [rat_tmp]
	mov	rsi, [rbx + 8 + rax * 8]
	call	rational_parse_cstr
	test	rax, rax
	jnz	.ppa_err
	mov	rdi, r12
	mov	rsi, r15
	mov	rdx, qword [rat_tmp + RATIONAL_NUM_OFF]
	mov	rcx, qword [rat_tmp + RATIONAL_DEN_OFF]
	call	poly_set_coeff_i64
	test	rax, rax
	jnz	.ppa_err
	inc	r15
	jmp	.ppa_loop
.ppa_done:
	xor	rax, rax
.ppa_err:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

usage:
	lea	rdi, [usage_msg]
	mov	rsi, usage_msg_len
	call	write_stderr
	exit 2

parse_error:
	lea	rdi, [parse_err_msg]
	mov	rsi, parse_err_msg_len
	call	write_stderr
	exit 2

too_many_error:
	lea	rdi, [too_many_msg]
	mov	rsi, too_many_msg_len
	call	write_stderr
	exit 2

write_stderr:
	mov	rdx, rsi
	mov	rsi, rdi
	mov	rdi, STDERR
	mov	rax, SYS_write
	syscall
	ret

cmd_frac db 'frac', 0
cmd_poly_derive db 'poly-derive', 0
cmd_poly_integrate db 'poly-integrate', 0
cmd_poly_eval db 'poly-eval', 0

op_add db 'add', 0
op_sub db 'sub', 0
op_mul db 'mul', 0
op_div db 'div', 0

usage_msg db 'usage: fmath frac <add|sub|mul|div> <a> <b>', 10
	db '       fmath poly-derive <c0> <c1> ...', 10
	db '       fmath poly-integrate <c0> <c1> ...', 10
	db '       fmath poly-eval <x> <c0> <c1> ...', 10
usage_msg_len = $ - usage_msg

parse_err_msg db 'fmath: parse/math error', 10
parse_err_msg_len = $ - parse_err_msg
too_many_msg db 'fmath: too many polynomial coefficients', 10
too_many_msg_len = $ - too_many_msg

segment readable writeable

argv_base dq ?
argc dq ?
poly_arg_count dq ?

rat_a rb RATIONAL_SIZE
rat_b rb RATIONAL_SIZE
rat_x rb RATIONAL_SIZE
rat_tmp rb RATIONAL_SIZE
rat_result rb RATIONAL_SIZE

poly_in rb POLY_SIZE
poly_out rb POLY_SIZE
poly_in_buf rb RATIONAL_SIZE * POLY_MAX_COEFFS
poly_out_buf rb RATIONAL_SIZE * POLY_WORK_CAP

include "fasm/core/runtime_bss.inc"
runtime_print_bss
