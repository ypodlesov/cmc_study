#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/types.h>

int 
myexec(char *cmd) 
{
    int pid = fork();
    if (pid < 0) {
        exit(1);
    } else if (!pid) {
        execlp(cmd, cmd, NULL);
        exit(1);
    } else {
        int status;
        wait(&status);
        return (WIFEXITED(status) && !WEXITSTATUS(status));
    }
    return 0;
}

int
main(int argc, char *argv[]) 
{
    return !((argc == 4) && (myexec(argv[1]) || myexec(argv[2])) && myexec(argv[3]));
}
