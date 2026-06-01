; OOP-style Playlist demo (vtable + self pointer).
;
; Like Python:
;   pl = Playlist()
;   pl.append(3); pl.append(1); ...
;   pl.print()
;   pl.reverse()
;   pl.print()

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/dynvec.inc"
include "fasm/core/listnode.inc"
include "fasm/core/print_value.inc"
include "fasm/core/leetcode.inc"
include "fasm/core/oop.inc"

entry start

start:
	lea	rdi, [storage_heap]
	call	heap_init

	lea	rdi, [storage_heap]
	call	playlist_new
	mov	[storage_playlist], rax

	; pl.append(3) — same idea as obj.method() via vtable
	mov	rdi, [storage_playlist]
	mov	rsi, PLAYLIST_VT_APPEND_OFF
	mov	rdx, 3
	call	playlist_invoke

	mov	rdi, [storage_playlist]
	mov	rsi, PLAYLIST_VT_APPEND_OFF
	mov	rdx, 1
	call	playlist_invoke

	mov	rdi, [storage_playlist]
	mov	rsi, PLAYLIST_VT_APPEND_OFF
	mov	rdx, 4
	call	playlist_invoke

	mov	rdi, [storage_playlist]
	mov	rsi, PLAYLIST_VT_APPEND_OFF
	mov	rdx, 1
	call	playlist_invoke

	mov	rdi, [storage_playlist]
	mov	rsi, PLAYLIST_VT_APPEND_OFF
	mov	rdx, 5
	call	playlist_invoke

	; pl.print()
	mov	rdi, [storage_playlist]
	mov	rsi, PLAYLIST_VT_PRINT_OFF
	xor	rdx, rdx
	call	playlist_invoke
	mov	al, 10
	call	print_char

	; pl.reverse()
	mov	rdi, [storage_playlist]
	mov	rsi, PLAYLIST_VT_REVERSE_OFF
	xor	rdx, rdx
	call	playlist_invoke

	mov	rdi, [storage_playlist]
	mov	rsi, PLAYLIST_VT_PRINT_OFF
	xor	rdx, rdx
	call	playlist_invoke
	mov	al, 10
	call	print_char

	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_SUCCESS

segment readable writeable

storage_heap rb HEAP_SIZE
storage_playlist dq ?

include "fasm/core/runtime_bss.inc"
runtime_value_bss
