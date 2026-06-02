class Logknife < Formula
  desc "Tiny native structured log slicer for plain logs and JSONL"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/logknife-0.1.0-macos-x86_64.tar.gz"
  sha256 "042ae9ed425a817aa5f7d5a9056f58e6ce6ca00767e56e3c333317b399ffe57a"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "logknife"
  end

  test do
    (testpath/"app.jsonl").write <<~EOS
      {"level":"info","msg":"ok"}
      {"level":"error","msg":"boom"}
    EOS
    assert_match '{"level":"error","msg":"boom"}', shell_output("#{bin}/logknife --jsonl --level error #{testpath}/app.jsonl")
  end
end
