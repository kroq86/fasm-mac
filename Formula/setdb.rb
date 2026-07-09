class Setdb < Formula
  desc "Tiny pure set-theoretic database CLI"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/setdb-0.1.0-macos-x86_64.tar.gz"
  sha256 "86ae6cc391dc6ce95547c2948a7d686a1264a50808a6ff07dd333dd2acc8bfeb"
  license "BSD-2-Clause"
  version "0.1.0"

  def install
    bin.install "setdb"
  end

  test do
    db = testpath/"universe.db"
    system bin/"setdb", "new", db
    system bin/"setdb", "add", db, "users", "alice", "bob"
    system bin/"setdb", "add", db, "admins", "alice"
    assert_equal "bob\n", shell_output("#{bin}/setdb diff #{db} users admins")
  end
end
