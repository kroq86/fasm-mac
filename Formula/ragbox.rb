class Ragbox < Formula
  desc "Local semantic snapshot CLI for agents (exact cosine search over .lv indexes)"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.3.0/ragbox-0.3.0-macos-x86_64.tar.gz"
  sha256 "b7443175638b1ab0ad59a1b85c773f7820e1830f95ba4f72a0c2d953debe6ad9"
  license "BSD-2-Clause"
  version "0.3.0"

  def caveats
    <<~EOS
      ragbox is an x86_64 binary. On Apple Silicon run via Rosetta:
        arch -x86_64 ragbox ...

      Text build/search requires Ollama with an embedding model (e.g. nomic-embed-text):
        brew install ollama && ollama pull nomic-embed-text
    EOS
  end

  def install
    bin.install "ragbox"
  end

  test do
    ragbox = bin/"ragbox"
    if Hardware::CPU.arm?
      assert_match "ollama checks skipped",
                   shell_output("arch -x86_64 #{ragbox} doctor --skip-ollama")
    else
      assert_match "ollama checks skipped", shell_output("#{ragbox} doctor --skip-ollama")
    end

    repo = testpath/"repo"
    (repo/"docs").mkpath
    (repo/"docs/auth.md").write("JWT authentication middleware validates bearer tokens.")

    manifest = testpath/"memory.lv.manifest.json"
    build_args = [
      "build",
      "--root", repo,
      "--out", testpath/"memory.lv",
      "--manifest", manifest,
      "--dry-run",
    ]
    if Hardware::CPU.arm?
      system "arch", "-x86_64", ragbox, *build_args
    else
      system ragbox, *build_args
    end
    assert_predicate manifest, :exist?

    records = JSON.parse(manifest.read)["records"]
    assert records.any? { |r| r["path"] == "docs/auth.md" }
  end
end
