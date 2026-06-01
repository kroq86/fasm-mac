; LeetCode 217. Contains Duplicate
; https://leetcode.com/problems/contains-duplicate/
; nums=[1,2,3,1]  =>  1

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/hashmap.inc"
include "fasm/examples/leetcode/common.inc"

entry start

start:
	lea	rdi, [storage_heap]
	call	heap_init
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, HASHMAP_DEFAULT_BUCKETS
	call	hashmap_init
	call	contains_duplicate
	call	print_bool_nl
	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_SUCCESS

contains_duplicate:
	push	rbx
	xor	rbx, rbx
.loop:
	cmp	rbx, test_nums_count
	jae	.no
	mov	rsi, [test_nums + rbx * 8]
	lea	rdi, [storage_map]
	call	hashmap_contains
	test	rax, rax
	jnz	.yes
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [test_nums + rbx * 8]
	mov	rcx, 1
	call	hashmap_put
	inc	rbx
	jmp	.loop
.yes:
	mov	rax, 1
	jmp	.done
.no:
	xor	rax, rax
.done:
	pop	rbx
	ret

segment readable writeable

storage_heap rb HEAP_SIZE
storage_map rb HASHMAP_SIZE
test_nums dq 1, 2, 3, 1
test_nums_count = 4

include "fasm/core/runtime_bss.inc"
runtime_print_bss
