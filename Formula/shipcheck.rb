class Shipcheck < Formula
  desc "Tiny native local release artifact QA checker"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/shipcheck-0.1.0-macos-x86_64.tar.gz"
  sha256 "bf0d40566977db7c269e86d1e0b8c7e7e2c3ef8e1c61cb96443ba970dc417aea"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "shipcheck"
  end

  test do
    assert_match "usage: shipcheck", shell_output("#{bin}/shipcheck 2>&1", 2)
  end
end
