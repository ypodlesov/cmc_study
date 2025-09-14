#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define handle_error(msg) do { perror(msg); exit(EXIT_FAILURE); } while (0)

enum 
{
    PAGE_SIZE = 512
};

int
main(int argc, char *argv[])
{
    char *path = strdup(argv[1]);
    int fd = open(path, O_RDONLY);
    struct stat stb;
    if (fd < 0) handle_error("open");
    if (fstat(fd, &stb)) handle_error("fstat");

    unsigned long tbl_offset = strtoul(argv[2], NULL, 16);
    unsigned short *vmem = mmap(NULL, stb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (vmem == MAP_FAILED) handle_error("mmap");

    unsigned short *pa_tbl = vmem + (unsigned long) tbl_offset / sizeof(*pa_tbl);
    unsigned long tmp;
    while (scanf("%lx", &tmp) == 1) {
        unsigned long pa_offset = tmp >> 9;
        printf("%u\n", vmem[(pa_tbl[pa_offset] + (tmp & (PAGE_SIZE - 1))) / sizeof (*vmem)]);
    }
    munmap(vmem, stb.st_size);
    close(fd);


    return 0;
}
