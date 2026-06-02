class Machodoctor < Formula
  desc "Tiny native macOS Mach-O binary inspector"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/machodoctor-0.1.0-macos-x86_64.tar.gz"
  sha256 "98695b3dc751025be81ad3a951804e471fab7783b3ae07bdaebbc61ce1dea025"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "machodoctor"
  end

  test do
    assert_match "format: Mach-O 64-bit", shell_output("#{bin}/machodoctor #{bin}/machodoctor")
  end
end
