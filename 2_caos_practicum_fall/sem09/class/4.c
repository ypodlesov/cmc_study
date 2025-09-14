#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdlib.h>

int
main()
{
    int pid = fork();
    if (pid < 0) abort();
    if (!pid) {
        int fd = open("/tmp/out.txt", O_WRONLY | O_TRUNC | O_CREAT, 0600);
        if (fd < 0) {
            _exit(1);
        }
        if (dup2(fd, STDOUT_FILENO) < 0) {
            _exit(1);
        }
        close(fd);
        if (chdir("/usr/bin") < 0) {
            _exit(1);
        }
        execlp("ls", "ls", "-l", NULL);
        _exit(1);
    } 
    wait(NULL);
}
