class Httpmini < Formula
  desc "Single-threaded concurrent static HTTP server in FASM"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/httpmini-0.1.0-macos-x86_64.tar.gz"
  sha256 "c8e0671b3b428027cac362d237a80764ce4da1d4c44d2c706f255b07aa293f6c"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "httpmini"
  end

  test do
    assert_match "usage: httpmini", shell_output("#{bin}/httpmini --bad 2>&1", 2)
  end
end
