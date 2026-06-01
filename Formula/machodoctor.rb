class Machodoctor < Formula
  desc "Tiny native macOS Mach-O binary inspector"
  homepage "https://github.com/kroq86/machodoctor"
  url "https://github.com/kroq86/machodoctor/releases/download/v0.1.0/machodoctor-0.1.0-macos-x86_64.tar.gz"
  sha256 "643887828f4879136933e4c97eec66bb54d43d3ee18c58b8416991fdc857a518"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "machodoctor"
  end

  test do
    assert_match "format: Mach-O 64-bit", shell_output("#{bin}/machodoctor #{bin}/machodoctor")
  end
end
