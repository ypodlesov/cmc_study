#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <fcntl.h>

enum 
{
    MODE = 0660
};

int
main(int argc, char *argv[])
{
    int status;
    int pid = fork();
    if (!pid) {
        if (argc < 5) {
            exit(42);
        }
        int fdin = open(argv[2], O_RDONLY);
        int fdout = open(argv[3], O_CREAT | O_APPEND | O_WRONLY, MODE);
        int fderr = open(argv[4], O_CREAT | O_TRUNC | O_WRONLY, MODE);
        if (fdout < 0 || fderr < 0 || fdin < 0) exit(42);
        dup2(fdin, 0);
        dup2(fdout, 1);
        dup2(fderr, 2);
        if (close(fdin) || close(fdout) || close(fderr)) exit(42);
        execlp(argv[1], argv[1], NULL);
        exit(42);
    }
    wait(&status);
    printf("%d\n", status);

    return 0;
}
