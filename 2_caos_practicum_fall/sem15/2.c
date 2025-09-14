#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>

int cur;
int mask[8] = {1 << 7, 1 << 6, 1 << 5, 1 << 4, 1 << 3, 1 << 2, 1 << 1, 1};
char byte;
int flag1;
int flag2 = 1;

void
handler1(int sig)
{
    if (sig != SIGUSR1 && sig != SIGUSR2 && sig != SIGIO) {
        printf("missed!\n");
        fflush(stdout);
        return;
    }
    if (sig == SIGUSR2) {
        byte |= mask[cur++];
        
    } else if (sig == SIGUSR1) {
        ++cur;
    }
    if (cur == 8) {
        printf("%c", byte);
        fflush(stdout);
        byte = 0;
        cur = 0;
    }
    kill(0, SIGALRM);
    if (sig == SIGIO) {
        flag1 = 1;
    }
}

void
handler2(int sig) 
{
    if (sig == SIGALRM) {
        flag2 = 1;
    }   
}

void 
proc1(void)
{   
    while (!flag1) {
        pause();
    }
    _exit(0);
}

void 
proc2(const char *path, int pid1) 
{
    sigaction(SIGALRM, &(struct sigaction) { .sa_handler = handler2, .sa_flags = SA_RESTART }, NULL);
    byte = 0;
    int fd = open(path, O_RDONLY);
    while (read(fd, &byte, sizeof(byte)) == sizeof(byte)) {
        for (int i = 0; i < CHAR_BIT; ++i) {
            flag2 = 0;
            kill(pid1, byte & mask[i] ? SIGUSR2 : SIGUSR1);
            while (!flag2) {
                pause();
            }
        }
        
    }
    close(fd);
    flag2 = 0;
    kill(pid1, SIGIO);
    while (!flag2) {
        pause();
    }
    fflush(stdout);
    _exit(0);
}

int
main(int argc, char *argv[]) {

    sigaction(SIGUSR1, &(struct sigaction) { .sa_handler = handler1, .sa_flags = SA_RESTART }, NULL);
    sigaction(SIGUSR2, &(struct sigaction) { .sa_handler = handler1, .sa_flags = SA_RESTART }, NULL);
    sigaction(SIGIO, &(struct sigaction) { .sa_handler = handler1, .sa_flags = SA_RESTART }, NULL);
    //sigaction(SIGALRM, &(struct sigaction) { .sa_handler = handler1, .sa_flags = SA_RESTART }, NULL);
    signal(SIGALRM, SIG_IGN);
    int pid1;
    if (!(pid1 = fork())) proc1();
    if (!fork()) proc2(argv[1], pid1);
    while(wait(NULL) > 0) {}

    return 0;
}





