; EML operator leaf (arXiv:2603.21852): eml(x,y) = exp(x) - log(y).
;
; Real f64 primitive for replaceable math backends. Full paper semantics use
; complex intermediates (see eml_sr C++ tool); this leaf matches the real-axis
; identity eml(x,1) = exp(x).
;
; Requires linked libm (exp, log). libm clobbers caller-saved XMM regs.

extrn exp
extrn log

EML_F64_ONE = 0x3ff0000000000000
EML_F64_TWO = 0x4000000000000000

; contract: xmm0 = x, xmm1 = y (must be > 0)
; returns: xmm0 = exp(x) - log(y)
; clobbers: xmm0-xmm1, rax, rsp (libm + scratch stack slots)
lb_eml_f64:
	movsd	xmm3, xmm0
	sub	rsp, 24
	movsd	[rsp + 16], xmm1
	movsd	xmm0, xmm3
	call	exp
	movsd	[rsp + 8], xmm0
	movsd	xmm0, [rsp + 16]
	call	log
	movsd	xmm1, xmm0
	movsd	xmm0, [rsp + 8]
	subsd	xmm0, xmm1
	add	rsp, 24
	ret

; contract: xmm0 = x
; returns: xmm0 = exp(x) via EML(x, 1)
; clobbers: xmm0-xmm1, xmm3, rax, rsp
lb_eml_exp_f64:
	mov	rax, EML_F64_ONE
	movq	xmm1, rax
	call	lb_eml_f64
	ret

; contract: xmm0 = x (must be > 0 for finite real result)
; returns: xmm0 = log(x) via EML(1, exp(EML(1, x)))
; clobbers: xmm0-xmm1, xmm3, rax, rsp
lb_eml_log_f64:
	sub	rsp, 24
	movsd	[rsp + 8], xmm0
	mov	rax, EML_F64_ONE
	movq	xmm0, rax
	movsd	xmm1, [rsp + 8]
	call	lb_eml_f64
	call	exp
	movsd	xmm1, xmm0
	mov	rax, EML_F64_ONE
	movq	xmm0, rax
	call	lb_eml_f64
	add	rsp, 24
	ret

; contract: xmm0 = a (must be > 0 for finite real result), xmm1 = b
; returns: xmm0 = a - b via EML(log(a), exp(b))
; clobbers: xmm0-xmm1, xmm3, rax, rsp
lb_eml_sub_f64:
	sub	rsp, 24
	movsd	[rsp + 8], xmm1
	call	lb_eml_log_f64
	movsd	[rsp + 16], xmm0
	movsd	xmm0, [rsp + 8]
	call	lb_eml_exp_f64
	movsd	xmm1, xmm0
	movsd	xmm0, [rsp + 16]
	call	lb_eml_f64
	add	rsp, 24
	ret

; contract: xmm0 = x
; returns: xmm0 = -x via sub(log(1), x)
; clobbers: xmm0-xmm1, xmm3, rax, rsp
lb_eml_neg_f64:
	sub	rsp, 24
	movsd	[rsp + 8], xmm0
	mov	rax, EML_F64_ONE
	movq	xmm0, rax
	call	lb_eml_log_f64
	movsd	xmm1, [rsp + 8]
	call	lb_eml_sub_f64
	add	rsp, 24
	ret

; contract: xmm0 = a, xmm1 = b
; returns: xmm0 = a + b via sub(a, neg(b))
; clobbers: xmm0-xmm1, xmm3, rax, rsp
lb_eml_add_f64:
	sub	rsp, 24
	movsd	[rsp + 8], xmm0
	movsd	xmm0, xmm1
	call	lb_eml_neg_f64
	movsd	xmm1, xmm0
	movsd	xmm0, [rsp + 8]
	call	lb_eml_sub_f64
	add	rsp, 24
	ret

; contract: xmm0 = x (must be > 0 for finite real result)
; returns: xmm0 = 1/x via exp(neg(log(x)))
; clobbers: xmm0-xmm1, xmm3, rax, rsp
lb_eml_inv_f64:
	call	lb_eml_log_f64
	call	lb_eml_neg_f64
	call	lb_eml_exp_f64
	ret

; contract: xmm0 = a, xmm1 = b
; returns: xmm0 = a * b
; clobbers: xmm0-xmm1, xmm3, rax, rsp
lb_eml_mul_f64:
	mulsd	xmm0, xmm1
	ret

; contract: xmm0 = a, xmm1 = b (must be nonzero for finite result)
; returns: xmm0 = a / b
; clobbers: xmm0-xmm1, xmm3, rax, rsp
lb_eml_div_f64:
	divsd	xmm0, xmm1
	ret

; contract: xmm0 = x
; returns: xmm0 = x / 2
; clobbers: xmm0-xmm1, xmm3, rax, rsp
lb_eml_half_f64:
	mov	rax, EML_F64_TWO
	movq	xmm1, rax
	call	lb_eml_div_f64
	ret
