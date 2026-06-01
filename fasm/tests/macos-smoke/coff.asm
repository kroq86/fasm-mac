format MS64 COFF

section '.text' code readable executable

public add_one
add_one:
	lea	eax,[ecx+1]
	ret
