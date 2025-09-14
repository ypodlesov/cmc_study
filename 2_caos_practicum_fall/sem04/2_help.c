#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static const mode_t MODE = 0666;

int 
main(int argc, char *argv[]) 
{
    int fd = open(argv[1], O_RDWR | O_TRUNC | O_CREAT, MODE);
    int t = 5;
    double a = 1.0;
    while (t--) {
        write(fd, &a, 8);
        a += 1.0;
    }

    return 0;
}
