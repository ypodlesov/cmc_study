// код с консультации: но там писал его я в том числе 

#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>

void
killall(int *a, int n)
{
    for (int i = 0; a[i] != -1 && i < n; ++i) {
        kill(a[i], SIGKILL);
        waitpid(a[i], NULL, 0);
    }
    _exit(1);
}

int 
main(int argc, char *argv[])
{
    int procs[argc];
    for (int i = 0; i < argc; ++i) {
        procs[i] = -1;
    }
    int fds[2] = { 0, 1 };
    for (int i = 1; i < argc; ++i) {
        int tmpfds[2] = { 0, 1 };
        if (i != argc - 1) {
            pipe2(tmpfds, O_CLOEXEC);
        }
        int pid = fork();
        procs[i-1] = pid;
        if (pid < 0) {
            killall(procs, argc - 1);
        }
        if (!pid) {
            dup2(fds[0], STDIN_FILENO);
            dup2(tmpfds[1], STDOUT_FILENO);
            execlp(argv[i], argv[i], NULL);
            _exit(1);
        }
        if (tmpfds[1] != 1) {
            close(tmpfds[1]);
        }
        if (fds[0] != 0) {
            close(fds[0]);
        }
        fds[0] = tmpfds[0];
    }
    for (int i = 0; i < argc - 1; ++i) {
        if (waitpid(procs[i], NULL, 0) < 0) {
            killall(procs, argc - 1);
        }
    }
    return 0;
}
