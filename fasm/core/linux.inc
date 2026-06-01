; Common Linux x86_64 syscall definitions for cross-built examples.

SYS_read equ 0
SYS_write equ 1
SYS_open equ 2
SYS_close equ 3
SYS_lseek equ 8
SYS_mmap equ 9
SYS_exit equ 60

STDIN equ 0
STDOUT equ 1
STDERR equ 2

EXIT_SUCCESS equ 0
EXIT_FAILURE equ 1

O_RDONLY equ 0
O_WRONLY equ 1
O_RDWR equ 2
O_CREAT equ 64
O_TRUNC equ 512
O_APPEND equ 1024

macro syscall1 number, a
{
	mov rax, number
	mov rdi, a
	syscall
}

macro syscall2 number, a, b
{
	mov rax, number
	mov rdi, a
	mov rsi, b
	syscall
}

macro syscall3 number, a, b, c
{
	mov rax, number
	mov rdi, a
	mov rsi, b
	mov rdx, c
	syscall
}

macro syscall4 number, a, b, c, d
{
	mov rax, number
	mov rdi, a
	mov rsi, b
	mov rdx, c
	mov r10, d
	syscall
}

macro syscall6 number, a, b, c, d, e, f
{
	mov rax, number
	mov rdi, a
	mov rsi, b
	mov rdx, c
	mov r10, d
	mov r8, e
	mov r9, f
	syscall
}

macro exit code
{
	mov rax, SYS_exit
	mov rdi, code
	syscall
}

macro open_file filename, flags, mode
{
	syscall3 SYS_open, filename, flags, mode
}

macro read_file fd, buffer, size
{
	syscall3 SYS_read, fd, buffer, size
}

macro write_file fd, buffer, size
{
	syscall3 SYS_write, fd, buffer, size
}

macro close_file fd
{
	syscall1 SYS_close, fd
}
