; Common BSS for runtime programs.

macro runtime_print_bss
{
	char_buf rb 1
	int_digits rb 21
}

macro runtime_value_bss
{
	runtime_print_bss
	nil_msg db 'null', 0
	nil_msg_len = 4
}
