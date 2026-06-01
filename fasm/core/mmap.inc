; Low-level mmap/munmap wrappers for heap allocation.

match =macos, TARGET_OS {
	SYS_mmap equ 020000C5h
	SYS_munmap equ 02000049h
	MAP_PRIVATE equ 0002h
	MAP_ANON equ 01000h
	PROT_READ equ 1
	PROT_WRITE equ 2
	PAGE_SIZE equ 4096
}

match =linux, TARGET_OS {
	SYS_mmap equ 9
	SYS_munmap equ 11
	MAP_PRIVATE equ 02h
	MAP_ANON equ 20h
	PROT_READ equ 1
	PROT_WRITE equ 2
	PAGE_SIZE equ 4096
}

; rax = payload pointer, 0 on failure
; rdx = requested size (rounded up to page)
mmap_alloc_raw:
	push	rdi
	push	rsi
	push	rdx
	mov	rdi, 0
	mov	rsi, rdx
	mov	rdx, PROT_READ or PROT_WRITE
	mov	r10, MAP_PRIVATE or MAP_ANON
	mov	r8, -1
	xor	r9, r9
	mov	rax, SYS_mmap
	syscall
	pop	rdx
	pop	rsi
	pop	rdi
	cmp	rax, -4096
	jae	.mmap_fail
	ret
.mmap_fail:
	xor	rax, rax
	ret

; rdi = ptr from mmap_alloc_raw, rdx = size passed to alloc
munmap_free_raw:
	mov	rsi, rdx
	mov	rax, SYS_munmap
	syscall
	ret

macro align_up reg, align
{
	.add = reg + align - 1
	mov reg, .add
	and reg, not (align - 1)
}
