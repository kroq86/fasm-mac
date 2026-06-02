class Fscan < Formula
  desc "Tiny native literal-search CLI for plain text files"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/fscan-0.1.0-macos-x86_64.tar.gz"
  sha256 "29e8eca3d3ef6e0ab3efa34b529016938666d69ca8cfe071930fcdcb4356b147"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "fscan"
  end

  test do
    (testpath/"a.txt").write("needle one\nplain\n")
    assert_equal "#{testpath}/a.txt:1", shell_output("#{bin}/fscan -c needle #{testpath}/a.txt").strip
  end
end
