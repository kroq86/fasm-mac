class FasmMac < Formula
  desc "flat assembler classic for macOS with Mach-O bridge"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.3.0/fasm-mac-0.3.0.tar.gz"
  sha256 "a2bbc51a03502ed3e96e5191e0fd1c9d63b2f67405a9c53008c8c7d45beb942b"
  license "BSD-2-Clause"
  version "0.3.0"

  depends_on "python@3.13"

  def install
    prefix.install "fasm"
    libexec.install "bin/fasm"
    inreplace libexec/"fasm",
               'PYTHON="${PYTHON:-python3}"',
               <<~'EOS'.chomp
                 PYTHON="${PYTHON:-python3}"
                 if [[ ! -x "$PYTHON" ]]; then
                   PYTHON="python3"
                 fi
               EOS
    bin.install_symlink libexec/"fasm"
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
