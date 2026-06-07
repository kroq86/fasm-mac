class Httpmini < Formula
  desc "Single-threaded concurrent static HTTP server in FASM"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/httpmini-0.1.0-macos-x86_64.tar.gz"
  sha256 "623f2f7bd6ab675c1de9c9b9580b9f7d9440513ef4856df54a6796ea8ab2d4ad"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "httpmini"
  end

  test do
    assert_match "usage: httpmini", shell_output("#{bin}/httpmini --bad 2>&1", 2)
  end
end
