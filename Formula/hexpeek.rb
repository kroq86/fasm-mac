class Hexpeek < Formula
  desc "Tiny native hex dump CLI for peeking at file bytes"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/hexpeek-0.1.0-macos-x86_64.tar.gz"
  sha256 "48b5d20580e4838be18403ae21b265202a496c1ae76476cb58483295f4dbfa65"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "hexpeek"
  end

  test do
    (testpath/"sample.bin").write("ABC\n")
    assert_match "00000000: 41 42 43 0a", shell_output("#{bin}/hexpeek -n 4 #{testpath}/sample.bin")
  end
end
