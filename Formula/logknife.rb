class Logknife < Formula
  desc "Tiny native structured log slicer for plain logs and JSONL"
  homepage "https://github.com/kroq86/logknife"
  url "https://github.com/kroq86/logknife/releases/download/v0.1.0/logknife-0.1.0-macos-x86_64.tar.gz"
  sha256 "7e3b636df5608642a851080a059e3854fc135c19b982399147cd93ea8563a7f3"
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
