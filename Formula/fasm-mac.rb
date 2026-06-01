class FasmMac < Formula
  desc "flat assembler classic for macOS with Mach-O bridge"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/fasm-mac-0.1.0.tar.gz"
  sha256 "df61a26fba6e945261f3fcfb7eded81883d8ab4e36ae7c192415719ac7cbfce4"
  license "BSD-2-Clause"
  version "0.1.0"

  depends_on "python@3.13"

  def install
    prefix.install "fasm"
    libexec.install "bin/fasm"
    (bin/"fasm").write_env_script libexec/"fasm",
                                  PYTHON: Formula["python@3.13"].opt_bin/"python3"
  end

  def caveats
    <<~EOS
      fasm-mac emits x86_64 Mach-O executables. On Apple Silicon, run output
      through Rosetta: arch -x86_64 ./program

      Use `fasm --emit=elf` for raw ELF output.
    EOS
  end

  test do
    platform_inc = prefix/"fasm/core/platform.inc"
    (testpath/"hello.asm").write <<~ASM
      format ELF64 executable 3
      include "#{platform_inc}"

      segment readable executable
      entry start

      start:
          write_file STDOUT, msg, msg_len
          exit 0

      segment readable writeable
      msg db "hello", 10
      msg_len = $ - msg
    ASM

    system bin/"fasm", testpath/"hello.asm"
    assert_predicate testpath/"hello", :exist?
    assert_match "Mach-O 64-bit", shell_output("file #{testpath}/hello")
  end
end
