#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

const char path[] = "fifo";

int
main(int argc, char *argv[]) 
{
    int serial = strtol(argv[1], NULL, 10);
    int pid = getpid();
    int fd = open(path, O_RDWR);
    FILE *fout = fdopen(fd, "a");
    fprintf(fout, "%d\n%d\n", serial, pid);
    fflush(fout);
    fclose(fout);
    close(fd);

    return 0;
}
