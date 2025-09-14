#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const mode_t MODE = 0600;

enum
{
    FIRST_SHIFT = 20,
    SECOND_SHIFT = 12,
    THIRD_SHIFT = 8,
    FIRST_BIT_MASK = 0x000000ff,
    SECOND_BIT_MASK = 0x00000f00,
    THIRD_BIT_MASK = 0x000ff000,
    FORTH_BIT_MASK = 0x00f00000,
    N_BYTES = 4
};

int 
main(int argc, char* argv[]) 
{
    if (argc < 1) {
        fprintf(stderr, "program: error! need at least 2 argument\n");
    }
    int fd = open(argv[1], O_WRONLY | O_TRUNC | O_CREAT, MODE);
    if (fd == -1) {
        fprintf(stderr, "program: erroe! cannot open file\n");
        exit(1);
    }
    unsigned int a; 
    while (scanf("%u", &a) == 1) {
        unsigned char buf[N_BYTES] = 
        {
            (a & FORTH_BIT_MASK) >> FIRST_SHIFT,
            (a & THIRD_BIT_MASK) >> SECOND_SHIFT,
            (a & SECOND_BIT_MASK) >> THIRD_SHIFT,
            a & FIRST_BIT_MASK
        };
        int wr_res = write(fd, &buf, N_BYTES);
        if (wr_res < N_BYTES) {
            fprintf(stderr, "program: syscall error!\n");
            exit(1);
        }
    }
    close(fd);

    return 0;
}












