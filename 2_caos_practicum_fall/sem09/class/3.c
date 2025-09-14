#include <stdio.h>
#include <unistd.h>

int
sys(const char *command)
{
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (!pid) {
        exec("/bin/sh", "sh", "-c", command, NULL);
        _exit(1);
    }
}
