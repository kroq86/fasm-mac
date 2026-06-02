class Pathsum < Formula
  desc "Tiny native recursive directory file and byte counter"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/pathsum-0.1.0-macos-x86_64.tar.gz"
  sha256 "ded7b4c5709557196fbea4bed3b0008461893d228a69f57610b20ebd097aa812"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "pathsum"
  end

  test do
    mkdir testpath/"tree"
    (testpath/"tree/a.txt").write("abc")
    output = shell_output("#{bin}/pathsum #{testpath}/tree")
    assert_match "files 1", output
    assert_match "bytes 3", output
  end
end
