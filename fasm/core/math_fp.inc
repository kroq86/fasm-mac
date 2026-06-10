; Floating-point math helpers: classification, rounding, modulo, frexp.
; Adapted from HeavyThing math.inc by Jeff Marrison (GPLv2+).
; https://2ton.com.au/HeavyThing/
;
; All operate on IEEE 754 double-precision (64-bit) values.
;
; fp_isnan      xmm0         → eax (1=NaN, 0=not)
; fp_isinfinite xmm0         → eax (1=±inf, 0=not)
; fp_isfinite   xmm0         → eax (1=finite, 0=not)
; fp_floor      xmm0         → xmm0
; fp_ceil       xmm0         → xmm0
; fp_fmod       xmm0 xmm1    → xmm0  (no error checking)
; fp_frexp      xmm0 rdi=exp → xmm0  (splits into mantissa + exponent)
;
; All preserve callee-saved registers. xmm0 is the only XMM clobbered
; except where noted.

; ---- constants (in readable data alongside these functions) ----
_fp_zero	dq  0.0
_fp_one		dq  1.0
_fp_negone	dq  -1.0
_fp_negzero	dq  0x8000000000000000			; -0.0
_fp_frexp_t54	dq  1.80143985094819840000e+16		; 2^54

; bitmask tables for floor mantissa clearing
_fp_bmask dw  0xffff, 0xfffe, 0xfffc, 0xfff8, 0xfff0, 0xffe0, 0xffc0
          dw  0xff80, 0xff00, 0xfe00, 0xfc00, 0xf800, 0xf000, 0xe000
          dw  0xc000, 0x8000, 0x0000

; ------------------------------------------------------------
; fp_isnan — xmm0 → eax (1 if NaN)
; IEEE 754: NaN iff exponent=0x7ff and mantissa≠0.
; ------------------------------------------------------------
fp_isnan:
	sub	rsp, 8
	movq	[rsp], xmm0
	mov	eax, [rsp+4]		; high dword
	mov	ecx, [rsp]		; low dword
	mov	edx, ecx
	and	eax, 0x7fffffff		; strip sign bit
	neg	edx
	mov	r8d, 0x7ff00000
	or	ecx, edx		; nonzero if mantissa != 0
	shr	ecx, 31
	or	eax, ecx
	sub	r8d, eax
	shr	r8d, 31
	mov	eax, r8d
	add	rsp, 8
	ret

; ------------------------------------------------------------
; fp_isinfinite — xmm0 → eax (1 if ±infinity)
; ------------------------------------------------------------
fp_isinfinite:
	sub	rsp, 8
	movq	[rsp], xmm0
	mov	eax, [rsp+4]		; high dword
	mov	ecx, [rsp]		; low dword
	mov	edx, eax
	and	eax, 0x7fffffff
	xor	eax, 0x7ff00000		; 0 iff exponent field == inf pattern
	or	ecx, eax
	mov	eax, ecx
	neg	eax
	or	ecx, eax
	shr	ecx, 31
	shr	edx, 30
	not	ecx
	and	ecx, edx
	mov	eax, ecx
	add	rsp, 8
	ret

; ------------------------------------------------------------
; fp_isfinite — xmm0 → eax (1 if finite, i.e. not inf and not NaN)
; ------------------------------------------------------------
fp_isfinite:
	call	fp_isinfinite
	xor	ecx, ecx
	test	eax, eax
	setz	cl
	mov	eax, ecx
	ret

; ------------------------------------------------------------
; fp_floor — xmm0 → xmm0 (round toward -∞)
; Works by zeroing the fractional mantissa bits below the binary point.
; ------------------------------------------------------------
fp_floor:
	sub	rsp, 16
	movq	[rsp], xmm0		; save value on stack for bit manipulation
	xorpd	xmm1, xmm1

	call	fp_isnan
	test	eax, eax
	jnz	.passthrough

	call	fp_isinfinite
	test	eax, eax
	jnz	.passthrough

	comisd	xmm0, xmm1
	je	.passthrough

	; extract biased exponent
	movzx	ecx, word [rsp+6]
	shr	ecx, 4
	and	ecx, 0x7ff
	mov	rax, rsp
	sub	ecx, 0x3ff		; unbiased exponent

	cmp	ecx, 0
	jge	.exp_nonneg

	; exponent < 0: value is in (-1, 1)
	comisd	xmm0, xmm1
	jae	.retzero		; 0 ≤ x < 1 → floor = 0
	movq	xmm0, [_fp_negone]
	add	rsp, 16
	ret
.retzero:
	xorpd	xmm0, xmm0
	add	rsp, 16
	ret

.exp_nonneg:
	; zero out the (52 - exponent) least significant mantissa bits
	mov	edx, 52
	sub	edx, ecx		; bits to clear
	mov	ecx, edx
.clear_loop:
	cmp	ecx, 16
	jb	.clear_partial
	mov	word [rax], 0
	add	rax, 2
	sub	ecx, 16
	jmp	.clear_loop
.clear_partial:
	test	ecx, ecx
	jle	.cleared
	movzx	edx, word [rax]
	lea	r9, [_fp_bmask]
	movzx	r8d, word [r9 + rcx*2]
	and	dx, r8w
	mov	word [rax], dx
.cleared:
	; if original was negative and we changed bits, subtract 1
	comisd	xmm0, xmm1
	jae	.retrsp
	movq	xmm1, [rsp]
	comisd	xmm0, xmm1
	je	.retrsp
	movq	xmm0, [rsp]
	subsd	xmm0, [_fp_one]
	add	rsp, 16
	ret
.retrsp:
	movq	xmm0, [rsp]
	add	rsp, 16
	ret
.passthrough:
	add	rsp, 16
	ret

; ------------------------------------------------------------
; fp_ceil — xmm0 → xmm0 (round toward +∞)
; ceil(x) = -floor(-x) for non-integer x, or x if already integer.
; Implemented as: floor(x) + 1 if floor(x) < x.
; ------------------------------------------------------------
fp_ceil:
	call	fp_isnan
	test	eax, eax
	jnz	.passthrough

	call	fp_isfinite
	test	eax, eax
	jz	.passthrough

	movq	xmm2, xmm0		; save x
	call	fp_floor		; xmm0 = floor(x)
	comisd	xmm0, xmm2
	jae	.no_add			; floor(x) >= x: already integer or negative
	addsd	xmm0, [_fp_one]
.no_add:
	; handle -0 case: if result is 0 and original was negative, return -0
	xorpd	xmm1, xmm1
	comisd	xmm0, xmm1
	jne	.done
	comisd	xmm2, xmm1
	jae	.done
	movq	xmm0, [_fp_negzero]
.done:
	ret
.passthrough:
	ret

; ------------------------------------------------------------
; fp_fmod — xmm0=x xmm1=y → xmm0 = x mod y
; No error checking. Result has same sign as x.
; fmod(x,y) = x - trunc(x/y)*y
; ------------------------------------------------------------
fp_fmod:
	movq	xmm2, xmm0
	divsd	xmm2, xmm1
	cvtsd2si rax, xmm2
	cvtsi2sd xmm2, rax
	mulsd	xmm2, xmm1
	subsd	xmm0, xmm2
	ret

; ------------------------------------------------------------
; fp_frexp — split double into normalised mantissa + exponent
; xmm0=x, rdi=*exp  → xmm0 = mantissa in [0.5, 1), *exp = exponent
; Sets *exp=0 and returns x unchanged for 0, NaN, Inf.
; ------------------------------------------------------------
fp_frexp:
	sub	rsp, 8
	movq	[rsp], xmm0
	mov	ecx, dword [rsp]	; lx (low 32 bits)
	mov	edx, dword [rsp+4]	; hx (high 32 bits)
	mov	eax, edx
	and	eax, 0x7fffffff
	mov	dword [rdi], 0		; *exp = 0
	cmp	eax, 0x7ff00000
	jae	.puke			; NaN or Inf: return unchanged
	mov	r8d, eax
	or	r8d, ecx
	jz	.puke			; zero: return 0
	cmp	eax, 0x00100000
	jb	.denormal
	; normal: exponent = ((hx >> 20) & 0x7ff) - 1022
	shr	eax, 20
	sub	eax, 1022
	mov	dword [rdi], eax
	and	edx, 0x800fffff
	or	edx, 0x3fe00000
	mov	dword [rsp+4], edx
	movq	xmm0, [rsp]
	add	rsp, 8
	ret
.denormal:
	; subnormal: multiply by 2^54 to normalise
	movq	xmm1, [_fp_frexp_t54]
	mulsd	xmm0, xmm1
	movq	[rsp], xmm0
	mov	edx, dword [rsp+4]
	mov	r8d, edx
	and	r8d, 0x7fffffff
	mov	dword [rdi], -54
	shr	r8d, 20
	sub	r8d, 1022
	add	dword [rdi], r8d
	and	edx, 0x800fffff
	or	edx, 0x3fe00000
	mov	dword [rsp+4], edx
	movq	xmm0, [rsp]
	add	rsp, 8
	ret
.puke:
	add	rsp, 8
	ret
