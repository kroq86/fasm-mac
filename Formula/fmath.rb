class Fmath < Formula
  desc "Tiny native exact math CLI for fractions and polynomials"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/fmath-0.1.0-macos-x86_64.tar.gz"
  sha256 "c2719f4ab227d3f74f1cdac2532fbdfcbfdfd0bab8edccb2c97350c684886097"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "fmath"
  end

  test do
    assert_equal "1/2\n", shell_output("#{bin}/fmath frac add 1/3 1/6")
    assert_equal "4x+3\n", shell_output("#{bin}/fmath poly-derive 1 3 2")
  end
end
