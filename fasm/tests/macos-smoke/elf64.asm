format ELF64 executable 3
entry start

segment readable executable

start:
	mov	eax,60
	xor	edi,edi
	syscall
