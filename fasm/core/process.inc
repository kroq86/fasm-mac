; Tiny libc-backed process helpers.
; Requires a linked libc. Callers must keep the System V stack aligned for C.

extrn system
extrn pipe
extrn posix_spawnp
extrn dup2
extrn close
extrn read
extrn waitpid

; contract: rdi = shell command cstr
; returns: rax = system() status
; clobbers: C ABI caller-saved registers
process_system:
	call	system
	ret

; contract: rdi = int fds[2]
; returns: rax = pipe() status
; clobbers: C ABI caller-saved registers
process_pipe:
	call	pipe
	ret

; contract: rdi = pid_t *pid, rsi = path, rdx = file_actions, rcx = attrs, r8 = argv, r9 = envp
; returns: rax = posix_spawnp() status
; clobbers: C ABI caller-saved registers
process_posix_spawnp:
	call	posix_spawnp
	ret

; contract: rdi = oldfd, rsi = newfd
; returns: rax = dup2() status
; clobbers: C ABI caller-saved registers
process_dup2:
	call	dup2
	ret

; contract: rdi = fd
; returns: rax = close() status
; clobbers: C ABI caller-saved registers
process_close:
	call	close
	ret

; contract: rdi = fd, rsi = buffer, rdx = count
; returns: rax = bytes read or -1
; clobbers: C ABI caller-saved registers
process_read:
	call	read
	ret

; contract: rdi = pid, rsi = status*, rdx = options
; returns: rax = waited pid or -1
; clobbers: C ABI caller-saved registers
process_waitpid:
	call	waitpid
	ret
