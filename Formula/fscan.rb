class Fscan < Formula
  desc "Tiny native literal-search CLI for plain text files"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/fscan-0.1.0-macos-x86_64.tar.gz"
  sha256 "3ad1e97cdda92f16819f7d7798a9f4adf024b16932101446aa6d6488e0d792a1"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "fscan"
  end

  test do
    (testpath/"a.txt").write("needle one\nplain\n")
    assert_equal "1", shell_output("#{bin}/fscan -c needle #{testpath}/a.txt").strip
  end
end
