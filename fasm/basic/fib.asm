format ELF64 executable 3
entry start

SYS_write equ 02000004h
SYS_exit equ 02000001h
STDOUT equ 1

segment readable executable

start:
	mov	r12,1
	mov	r13,1
	mov	r14,10

print_loop:
	mov	rdi,r12
	call	print_u64

	mov	rax,r12
	add	rax,r13
	mov	r12,r13
	mov	r13,rax

	dec	r14
	jnz	print_loop

	mov	eax,SYS_exit
	xor	edi,edi
	syscall

print_u64:
	lea	rsi,[number_buffer+31]
	mov	byte [rsi],10
	mov	rcx,10

convert_digit:
	xor	edx,edx
	mov	rax,rdi
	div	rcx
	add	dl,'0'
	dec	rsi
	mov	[rsi],dl
	mov	rdi,rax
	test	rax,rax
	jnz	convert_digit

	mov	eax,SYS_write
	mov	edi,STDOUT
	lea	rdx,[number_buffer+32]
	sub	rdx,rsi
	syscall
	ret

segment readable writeable

number_buffer rb 32
