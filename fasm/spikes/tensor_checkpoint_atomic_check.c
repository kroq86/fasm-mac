#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { PAYLOAD = 4096 };

static int write_all(int fd, const unsigned char *p, size_t n, long fail_after) {
    size_t done = 0;
    while (done < n) {
        size_t chunk = n - done;
        if (fail_after >= 0 && done + chunk > (size_t)fail_after) chunk = (size_t)fail_after - done;
        if (!chunk) { errno = EIO; return -1; }
        ssize_t wrote = write(fd, p + done, chunk);
        if (wrote <= 0) return -1;
        done += (size_t)wrote;
    }
    return 0;
}

static int sync_parent(const char *path) {
    char parent[1024];
    size_t n = strlen(path);
    if (n >= sizeof parent) return -1;
    memcpy(parent, path, n + 1);
    char *slash = strrchr(parent, '/');
    if (slash) *slash = '\0'; else strcpy(parent, ".");
    int fd = open(*parent ? parent : "/", O_RDONLY);
    if (fd < 0) return -1;
    int result = fsync(fd);
    if (close(fd)) result = -1;
    return result;
}

/* Same-directory temp + fsync + rename is the contract under test.  The
   caller-provided payload has already passed format/checksum construction. */
static int atomic_save(const char *path, const unsigned char *payload, size_t n, long fail_after) {
    char temp[1024];
    if (snprintf(temp, sizeof temp, "%s.tmp.XXXXXX", path) >= (int)sizeof temp) return -1;
    int fd = mkstemp(temp);
    if (fd < 0) return -1;
    int ok = write_all(fd, payload, n, fail_after) == 0 && fsync(fd) == 0 && close(fd) == 0;
    if (!ok) {
        int saved = errno;
        close(fd);
        unlink(temp);
        errno = saved;
        return -1;
    }
    if (rename(temp, path) || sync_parent(path)) {
        int saved = errno;
        unlink(temp);
        errno = saved;
        return -1;
    }
    return 0;
}

static int read_exact(const char *path, unsigned char *out, size_t n) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    size_t done = 0;
    while (done < n) {
        ssize_t got = read(fd, out + done, n - done);
        if (got <= 0) { close(fd); return -1; }
        done += (size_t)got;
    }
    unsigned char extra;
    int exact = read(fd, &extra, 1) == 0;
    close(fd);
    return exact ? 0 : -1;
}

static int temp_exists(const char *path) {
    char pattern[1024];
    if (snprintf(pattern, sizeof pattern, "%s.tmp.*", path) >= (int)sizeof pattern) return 1;
    glob_t matches = {0};
    int result = glob(pattern, 0, NULL, &matches);
    int exists = result == 0 && matches.gl_pathc != 0;
    globfree(&matches);
    return exists;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    unsigned char old[PAYLOAD], next[PAYLOAD], seen[PAYLOAD];
    for (int i = 0; i < PAYLOAD; i++) { old[i] = (unsigned char)(i * 17); next[i] = (unsigned char)(255 - i * 29); }
    if (atomic_save(argv[1], old, sizeof old, -1) || read_exact(argv[1], seen, sizeof seen) || memcmp(old, seen, sizeof old)) return 3;
    if (!atomic_save(argv[1], next, sizeof next, 137) || read_exact(argv[1], seen, sizeof seen) || memcmp(old, seen, sizeof old)) return 4;
    if (atomic_save(argv[1], next, sizeof next, -1) || read_exact(argv[1], seen, sizeof seen) || memcmp(next, seen, sizeof next)) return 5;
    if (temp_exists(argv[1])) return 6;
    puts("tensor checkpoint atomic spike passed: interrupted_write=old_preserved successful_write=atomic_replace file_fsync=yes directory_fsync=yes temp_cleanup=yes");
    return 0;
}
