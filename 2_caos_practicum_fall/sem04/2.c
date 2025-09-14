#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>


enum 
{
    DOUBLE_SZ = 8
};

int 
main(int argc, char *argv[]) 
{
    if (argc < 3) {
        fprintf(stderr, "program: error! expected at least 3 arguments\n");
        exit(1);
    }
    int fd = open(argv[1], O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "program: error!\n");
    }
    errno = 0;
    char *endptr = NULL;
    char *str = argv[2];
    long n = strtol(str, &endptr, 10);
    if (errno || *endptr || endptr == str || (int) n != n) {
        fprintf(stderr, "program: error!\n");
        exit(1);
    }

    int file_end = lseek(fd, 0, SEEK_END);
    int cur_pos = lseek(fd, 0, SEEK_SET);
    double prev = 0.0;
    cur_pos = lseek(fd, 0, SEEK_SET);
    while (cur_pos < file_end &&  cur_pos < n * DOUBLE_SZ) {
        double b;
        if (read(fd, &b, DOUBLE_SZ) < DOUBLE_SZ) {
            fprintf(stderr, "program: error!\n");
            exit(2);
        }
        b -= prev;
        prev = b;
        lseek(fd, cur_pos, SEEK_SET);
        write(fd, &b, DOUBLE_SZ);
        cur_pos += DOUBLE_SZ;
    }

    if (close(fd) == -1) {
        fprintf(stderr, "program: error!\n");
        exit(1);
    }


    return 0;
}
