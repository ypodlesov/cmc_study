#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>

const char path[] = "fifo";

int 
main(int argc, char *argv[]) 
{

    mkfifo(path, S_IFIFO | 0666);
    char *const av1[] = {"p2", "1", NULL};
    if (!fork()) {
        execv("p2", av1);
        exit(1);
    }
    int fd = open(path, O_RDONLY);
    FILE *fin = fdopen(fd, "r");
    int pid, serial;
    while (fscanf(fin, "%d%d", &serial, &pid) != EOF) {
        printf("Server %d got message from %d process with serial %d\n", getppid(), pid, serial);
        fflush(stdout);
    }
    fclose(fin);
    unlink(path);

    return 0;
}
