; Platform syscall layer for small runnable examples.
;
; The fasm-mac shim passes `-d TARGET_OS=macos` for Mach-O output and
; `-d TARGET_OS=linux` for ELF output. Use one of those values when invoking
; the upstream fasm host directly.

match =macos, TARGET_OS {
	SYS_exit equ 02000001h
	SYS_read equ 02000003h
	SYS_write equ 02000004h
	SYS_open equ 02000005h
	SYS_close equ 02000006h
	SYS_mkdir equ 02000088h
	SYS_fcntl equ 0200005Ch
	SYS_lseek equ 020000C7h
	SYS_socket equ 02000061h
	SYS_connect equ 02000062h
	SYS_accept equ 0200001Eh
	SYS_bind equ 02000068h
	SYS_listen equ 0200006Ah
	SYS_setsockopt equ 02000069h
	SYS_stat64 equ 02000152h
	SYS_fstat64 equ 02000153h
	SYS_lstat64 equ 02000154h
	SYS_fstat equ 020000BDh
	SYS_getdirentries64 equ 02000158h
	SYS_kqueue equ 0200016Ah
	SYS_kevent equ 0200016Bh

	O_RDONLY equ 0000h
	O_WRONLY equ 0001h
	O_RDWR equ 0002h
	O_NONBLOCK equ 0004h
	O_CREAT equ 0200h
	O_TRUNC equ 0400h
	O_APPEND equ 0008h
	SEEK_SET equ 0
	SEEK_END equ 2
	F_GETFL equ 3
	F_SETFL equ 4
	F_GETPATH equ 50
	EAGAIN equ 35

	AF_INET equ 2
	SOCK_STREAM equ 1
	IPPROTO_TCP equ 6
	SOL_SOCKET equ 0FFFFh
	SO_REUSEADDR equ 4
	INADDR_ANY equ 0
	SOCKADDR_IN_SIZE equ 16

	S_IFMT equ 0F000h
	S_IFDIR equ 04000h
	S_IFREG equ 08000h
	S_IFLNK equ 0A000h
	DT_DIR equ 4
	DT_REG equ 8
	DT_LNK equ 10
	STAT64_SIZE equ 144
	STAT64_MODE_OFF equ 4
	STAT64_SIZE_OFF equ 96
	DIRENT_RECLEN_OFF equ 16
	DIRENT_TYPE_OFF equ 20
	DIRENT_NAME_OFF equ 21

	EVFILT_READ equ -1
	EVFILT_WRITE equ -2
	EV_ADD equ 0001h
	EV_ONESHOT equ 0010h
	KEVENT_SIZE equ 32
	KEVENT_IDENT_OFF equ 0
	KEVENT_FILTER_OFF equ 8
	KEVENT_FLAGS_OFF equ 10
	KEVENT_FFLAGS_OFF equ 12
	KEVENT_DATA_OFF equ 16
	KEVENT_UDATA_OFF equ 24
}

match =linux, TARGET_OS {
	SYS_read equ 0
	SYS_write equ 1
	SYS_open equ 2
	SYS_close equ 3
	SYS_mkdir equ 83
	SYS_lseek equ 8
	SYS_exit equ 60

	O_RDONLY equ 0
	O_WRONLY equ 1
	O_RDWR equ 2
	O_CREAT equ 64
	O_TRUNC equ 512
	O_APPEND equ 1024
	SEEK_SET equ 0
	SEEK_END equ 2
}

STDIN equ 0
STDOUT equ 1
STDERR equ 2

EXIT_SUCCESS equ 0
EXIT_FAILURE equ 1

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

macro jump_if_syscall_error target
{
	match =macos, TARGET_OS \{
		jc target
	\}
	match =linux, TARGET_OS \{
		test rax, rax
		js target
	\}
}

; rax = 16-bit port in host order; returns ax in network byte order
macro htons port_reg
{
	mov ax, port_reg
	xchg ah, al
}
