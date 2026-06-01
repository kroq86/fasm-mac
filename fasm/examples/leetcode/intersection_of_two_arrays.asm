; LeetCode 349. Intersection of Two Arrays
; https://leetcode.com/problems/intersection-of-two-arrays/
; nums1=[1,2,2,1], nums2=[2,2]  =>  2

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
	xor	rbx, rbx
.load:
	cmp	rbx, nums1_count
	jae	.scan
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [nums1 + rbx * 8]
	mov	rcx, 1
	call	hashmap_put
	inc	rbx
	jmp	.load
.scan:
	xor	rbx, rbx
	xor	r12, r12
.scan_loop:
	cmp	rbx, nums2_count
	jae	.print
	mov	rsi, [nums2 + rbx * 8]
	lea	rdi, [storage_map]
	call	hashmap_get
	cmp	rax, 1
	jne	.scan_next
	mov	rax, [nums2 + rbx * 8]
	mov	[result + r12 * 8], rax
	inc	r12
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [nums2 + rbx * 8]
	mov	rcx, 2
	call	hashmap_put
.scan_next:
	inc	rbx
	jmp	.scan_loop
.print:
	lea	rdi, [result]
	mov	rsi, r12
	call	print_i64_array
	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_SUCCESS

segment readable writeable

storage_heap rb HEAP_SIZE
storage_map rb HASHMAP_SIZE
nums1 dq 1, 2, 2, 1
nums1_count = 4
nums2 dq 2, 2
nums2_count = 2
result rq 4

include "fasm/core/runtime_bss.inc"
runtime_print_bss
