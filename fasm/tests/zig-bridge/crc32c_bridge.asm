; C ABI wrapper for FASM CRC32C hot path.

format ELF64

section ".text" executable

public lb_crc32c

; uint32_t lb_crc32c(const uint8_t *ptr, uint64_t len)
; rdi = ptr, rsi = len, eax = CRC32C
lb_crc32c:
	call	crc32c_compute
	ret

include "fasm/core/crc32c.inc"
