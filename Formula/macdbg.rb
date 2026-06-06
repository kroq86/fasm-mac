class Macdbg < Formula
  desc "AI-native LLDB snapshot debugger for macOS binaries"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/macdbg-0.1.0-macos-x86_64.tar.gz"
  sha256 "4629fd366baa533f69e868ca7d3d4dbe2d239e4532a05ab6e9de69516ef5f744"
  license "BSD-2-Clause"
  version "0.1.0"

  depends_on arch: :x86_64
  depends_on "raylib"

  def install
    bin.install "macdbg"
    system "install_name_tool", "-add_rpath", Formula["raylib"].opt_lib, bin/"macdbg"
  end

  def caveats
    <<~EOS
      macdbg is an x86_64 Mach-O binary because fasm-mac currently emits
      x86_64 output. Install and run it under an x86_64/Rosetta Homebrew
      environment so the raylib dependency has the same architecture.
    EOS
  end

  test do
    assert_match "usage: macdbg", shell_output("#{bin}/macdbg --help")
  end
end
