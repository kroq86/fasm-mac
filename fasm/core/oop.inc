; Minimal "class" pattern: object + vtable + methods (self in rdi).
;
; Playlist wraps a linked list (ListNode) and exposes append/print/reverse
; like Python methods, via function pointers in a vtable.

struc Playlist
{
	.vtable dq ?
	.heap dq ?
	.head dq ?
	.count dq ?
}

virtual at 0
	_Playlist Playlist
	PLAYLIST_SIZE = $
end virtual

PLAYLIST_VTABLE_OFF equ 0
PLAYLIST_HEAP_OFF equ 8
PLAYLIST_HEAD_OFF equ 16
PLAYLIST_COUNT_OFF equ 24

PLAYLIST_VT_APPEND_OFF equ 0
PLAYLIST_VT_PRINT_OFF equ 8
PLAYLIST_VT_REVERSE_OFF equ 16

playlist_vtable:
	dq	playlist_append
	dq	playlist_print
	dq	playlist_reverse

; rdi = Heap*
; rax = Playlist*
playlist_new:
	push	r12
	mov	r12, rdi
	mov	rdi, r12
	mov	rsi, PLAYLIST_SIZE
	call	heap_alloc
	mov	rbx, rax
	lea	rcx, [playlist_vtable]
	mov	[rbx + PLAYLIST_VTABLE_OFF], rcx
	mov	[rbx + PLAYLIST_HEAP_OFF], r12
	mov	qword [rbx + PLAYLIST_HEAD_OFF], 0
	mov	qword [rbx + PLAYLIST_COUNT_OFF], 0
	mov	rax, rbx
	pop	r12
	ret

; rdi = Playlist*, rsi = int64 value
playlist_append:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, [r12 + PLAYLIST_HEAP_OFF]
	mov	rsi, [r12 + PLAYLIST_HEAD_OFF]
	mov	rdx, r13
	call	list_push_back
	mov	[r12 + PLAYLIST_HEAD_OFF], rax
	inc	qword [r12 + PLAYLIST_COUNT_OFF]
	pop	r13
	pop	r12
	ret

; rdi = Playlist*
playlist_print:
	push	r12
	mov	r12, rdi
	mov	rdi, [r12 + PLAYLIST_HEAD_OFF]
	call	list_print
	pop	r12
	ret

; rdi = Playlist*
playlist_reverse:
	push	r12
	mov	r12, rdi
	mov	rdi, [r12 + PLAYLIST_HEAD_OFF]
	call	list_reverse
	mov	[r12 + PLAYLIST_HEAD_OFF], rax
	pop	r12
	ret

; Call a vtable method: rdi=Playlist*, rsi=vtable offset, rdx=optional arg
; For append: rdx=value. For print/reverse: rdx ignored.
playlist_invoke:
	mov	rax, [rdi + PLAYLIST_VTABLE_OFF]
	mov	rax, [rax + rsi]
	mov	rsi, rdx
	call	rax
	ret

; rdi = Playlist*
gc_mark_playlist:
	push	r12
	mov	r12, rdi
	mark_block r12
	mov	rdi, [r12 + PLAYLIST_HEAD_OFF]
	call	gc_mark_list
	pop	r12
	ret
