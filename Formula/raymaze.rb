class Raymaze < Formula
  desc "Tiny native raylib raycaster maze game"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/raymaze-0.1.0-macos-x86_64.tar.gz"
  sha256 "2586af4c0c268893cf06022d7b68eb0a7a48f0b9bc823c3dc8ae3c809bea295f"
  license "BSD-2-Clause"
  version "0.1.0"

  depends_on arch: :x86_64
  depends_on "raylib"

  def install
    bin.install "raymaze"
    system "install_name_tool", "-add_rpath", Formula["raylib"].opt_lib, bin/"raymaze"
  end

  def caveats
    <<~EOS
      raymaze is an x86_64 Mach-O binary because fasm-mac currently emits
      x86_64 output. Install and run it under an x86_64/Rosetta Homebrew
      environment so the raylib dependency has the same architecture.
    EOS
  end

  test do
    system bin/"raymaze", "--snapshot", testpath/"snapshot.ppm"
    assert_match "P6\n160 100\n255\n", (testpath/"snapshot.ppm").read[0, 15]
  end
end
