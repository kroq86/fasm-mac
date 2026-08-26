#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return(uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
static uint32_t checksum(const unsigned char*p,size_t n){uint32_t h=2166136261u;for(size_t i=0;i<n;i++){h^=p[i];h*=16777619u;}return h;}

static int make_file(const char*path,size_t n,uint32_t*expected){
    int fd=open(path,O_CREAT|O_TRUNC|O_WRONLY,0600);if(fd<0)return-1;
    unsigned char block[16384];size_t done=0;uint32_t h=2166136261u;
    while(done<n){size_t count=n-done<sizeof block?n-done:sizeof block;for(size_t i=0;i<count;i++)block[i]=(unsigned char)((done+i)*131u+17u);
        for(size_t i=0;i<count;i++){h^=block[i];h*=16777619u;}if(write(fd,block,count)!=(ssize_t)count){close(fd);return-1;}done+=count;}
    if(fsync(fd)||close(fd))return-1;*expected=h;return 0;
}

static int read_copy(const char*path,size_t n,uint32_t expected,uint64_t*elapsed){
    int fd=open(path,O_RDONLY);unsigned char*p=malloc(n);if(fd<0||!p){if(fd>=0)close(fd);free(p);return-1;}uint64_t start=now_ns();size_t done=0;
    while(done<n){ssize_t got=read(fd,p+done,n-done);if(got<=0){close(fd);free(p);return-1;}done+=(size_t)got;}
    uint32_t got=checksum(p,n);*elapsed=now_ns()-start;close(fd);free(p);return got==expected?0:-1;
}

static int mapped(const char*path,size_t n,size_t select,uint32_t expected,uint64_t*elapsed,uint32_t*selected_hash){
    int fd=open(path,O_RDONLY);if(fd<0)return-1;uint64_t start=now_ns();const unsigned char*p=mmap(NULL,n,PROT_READ,MAP_PRIVATE,fd,0);if(p==MAP_FAILED){close(fd);return-1;}
    uint32_t got=checksum(p,n);unsigned char*copy=malloc(select);if(!copy){munmap((void*)p,n);close(fd);return-1;}memcpy(copy,p+n-select,select);
    if(munmap((void*)p,n)){free(copy);close(fd);return-1;}*selected_hash=checksum(copy,select);*elapsed=now_ns()-start;free(copy);close(fd);return got==expected?0:-1;
}

int main(int argc,char**argv){if(argc!=2)return 2;const size_t sizes[]={4096,1u<<20,64u<<20};
    for(unsigned s=0;s<3;s++){uint32_t expected,mapped_tail,tail_expected;uint64_t read_ns,map_ns;size_t n=sizes[s],select=n/10? n/10:1;
        if(make_file(argv[1],n,&expected)||read_copy(argv[1],n,expected,&read_ns))return 3;
        int fd=open(argv[1],O_RDONLY);unsigned char*tail=malloc(select);if(fd<0||!tail||lseek(fd,(off_t)(n-select),SEEK_SET)<0||read(fd,tail,select)!=(ssize_t)select)return 4;
        tail_expected=checksum(tail,select);free(tail);close(fd);if(mapped(argv[1],n,select,expected,&map_ns,&mapped_tail)||mapped_tail!=tail_expected)return 5;
        printf("checkpoint mmap size=%zu read_copy_ms=%.3f mmap_verify_ms=%.3f copied_bytes=%zu->%zu copy_reduction=%.1fx correctness=exact\n",n,read_ns/1e6,map_ns/1e6,n,select,(double)n/select);
    }
    puts("tensor checkpoint mmap spike passed: modes=read+copy,mmap-readonly,mmap+selective-copy sizes=4KiB,1MiB,64MiB timing=informational");return 0;}
