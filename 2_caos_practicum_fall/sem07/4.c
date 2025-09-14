#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#define abs(A) ((A) > 0 ? (A) : -(A))

int
main(int argc, char *argv[]) 
{
    if (argc < 2) return 0;
    char *path = strdup(argv[1]);
    int fd;
    if ((fd = open(path, O_RDWR)) == -1) {
        fprintf(stderr, "error! cannot open file\n");
        exit(1);
    }
    struct stat stb;
    if (fstat(fd, &stb)) {
        fprintf(stderr, "error! fstat\n");
        exit(1);
    }
    int x;
    while (scanf("%d", &x) == 1) {
        if (x > 0) {
            if (stb.st_size * 8 >= x) { 
                lseek(fd, (x - 1) / 8, SEEK_SET);
                unsigned char val;
                read(fd, &val, sizeof(val));
                val |= 1U << ((x - 1) % 8);
                lseek(fd, (x - 1) / 8, SEEK_SET);
                write(fd, &val, sizeof(val));
            }
        } else if (x < 0) {
            x *= -1;
            if (stb.st_size * 8 >= x) { 
                lseek(fd, (x - 1) / 8, SEEK_SET);
                unsigned char val;
                read(fd, &val, sizeof(val));
                val &= ~(1U << ((x - 1) % 8));
                lseek(fd, (x - 1) / 8, SEEK_SET);
                write(fd, &val, sizeof(val));
            }
        } 
    }


}
