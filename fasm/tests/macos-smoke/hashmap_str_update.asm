; Regression: replacement preserves entry pointers across str_eq.
; Two colliding keys exercise both chain traversal and head/tail replacement.
format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable
include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/hashmap_str.inc"

macro put_int key_label, number
{
        lea rdi, [storage_heap]
        lea rsi, [storage_map]
        lea rdx, [key_label]
        mov rcx, number
        call hashmap_str_put_int
}

macro put_string key_label, value_label
{
        lea rdi, [storage_heap]
        lea rsi, [storage_map]
        lea rdx, [key_label]
        lea rcx, [value_label]
        call hashmap_str_put_str
}

macro expect_int key_label, number
{
        lea rdi, [storage_map]
        lea rsi, [key_label]
        call hashmap_str_get_int
        cmp rax, number
        jne failed
}

macro expect_string key_label, value_label
{
        lea rdi, [storage_map]
        lea rsi, [key_label]
        call hashmap_str_get_entry
        test rax, rax
        jz failed
        cmp qword [rax + HASH_ENTRY_STR_TYPE_OFF], HM_VAL_STR
        jne failed
        mov rdi, [rax + HASH_ENTRY_STR_VAL_OFF]
        lea rsi, [value_label]
        call str_eq
        test rax, rax
        jz failed
}

macro expect_size number
{
        cmp qword [storage_map + HASHMAP_STR_SIZE_OFF], number
        jne failed
}

entry start
start:
        lea rdi, [storage_heap]
        call heap_init
        lea rdi, [storage_heap]
        lea rsi, [storage_map]
        mov rdx, HASHMAP_DEFAULT_BUCKETS
        call hashmap_str_init

        put_int key_foo, 42
        expect_int key_foo, 42
        expect_size 1
        put_int key_foo, 84
        expect_int key_foo, 84
        expect_size 1

        ; Assert the fixture really collides under the current hash function.
        lea rdi, [key_foo]
        mov rsi, HASHMAP_DEFAULT_BUCKETS
        call hash_index_str
        mov r12, rax
        lea rdi, [key_collision]
        mov rsi, HASHMAP_DEFAULT_BUCKETS
        call hash_index_str
        cmp rax, r12
        jne failed

        put_int key_collision, 7
        expect_int key_collision, 7
        expect_int key_foo, 84
        expect_size 2
        put_int key_collision, 9
        put_int key_foo, 99
        expect_int key_collision, 9
        expect_int key_foo, 99
        expect_size 2

        put_string key_foo, value_first
        expect_string key_foo, value_first
        expect_int key_foo, HASHMAP_STR_WRONGTYPE
        expect_int key_collision, 9
        expect_size 2
        put_string key_foo, value_second
        expect_string key_foo, value_second
        put_int key_foo, 123
        expect_int key_foo, 123
        put_string key_collision, value_first
        expect_string key_collision, value_first
        expect_int key_foo, 123
        expect_size 2

        lea rdi, [storage_heap]
        call heap_destroy
        lea rdi, [passed_message]
        call print_cstr
        exit EXIT_SUCCESS
failed:
        lea rdi, [storage_heap]
        call heap_destroy
        exit EXIT_FAILURE

segment readable writeable
storage_heap rb HEAP_SIZE
storage_map rb HASHMAP_STR_SIZE
key_foo db 'foo', 0
key_collision db 'bff', 0
value_first db 'first', 0
value_second db 'second', 0
passed_message db 'hashmap_str update checks passed', 10, 0
include "fasm/core/runtime_bss.inc"
runtime_print_bss
