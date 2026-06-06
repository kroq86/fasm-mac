; Tiny libc-backed process helpers.
; Requires a linked libc. Callers must keep the System V stack aligned for C.

extrn system
extrn pipe
extrn posix_spawnp
extrn dup2
extrn close
extrn read
extrn waitpid

; rdi = shell command cstr; rax = system() status
process_system:
	call	system
	ret

; rdi = int fds[2]; rax = pipe() status
process_pipe:
	call	pipe
	ret

; rdi = pid_t *pid, rsi = path, rdx = file_actions, rcx = attrs,
; r8 = argv, r9 = envp; rax = posix_spawnp() status
process_posix_spawnp:
	call	posix_spawnp
	ret

; rdi = oldfd, rsi = newfd; rax = dup2() status
process_dup2:
	call	dup2
	ret

; rdi = fd; rax = close() status
process_close:
	call	close
	ret

; rdi = fd, rsi = buffer, rdx = count; rax = bytes read or -1
process_read:
	call	read
	ret

; rdi = pid, rsi = status*, rdx = options; rax = waited pid or -1
process_waitpid:
	call	waitpid
	ret
