#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

#define GETSERIAL(pid) ((pid) == pid1 ? 1 : 2)
#define OTHERPID(pid) ((pid) == pid1 ? pid2 : pid1)

unsigned long long m;
FILE *fin;
FILE *fout;

void
handler(int sig) 
{
    int curpid = getpid();
    int pid1, pid2;
    unsigned long long val;
    fscanf(fin, "%d%d%llu", &pid1, &pid2, &val);
    if (val < m) {
        printf("%d %llu\n", GETSERIAL(curpid), val);
        fflush(stdout);
    }
    fprintf(fout, "%d\n%d\n%llu\n", pid1, pid2, ++val);
    fflush(fout);
    kill(OTHERPID(curpid), SIGUSR1);
    if (val >= m) {
        fclose(fin);
        fclose(fout);
        _exit(0);
    }
}

int
main(int argc, char *argv[])
{
    m = strtoull(argv[1], NULL, 10);
    sigaction(SIGUSR1, &(struct sigaction){ .sa_handler = handler, .sa_flags = SA_RESTART }, NULL);
    int fd[2];
    pipe(fd);
    fin = fdopen(fd[0], "r");
    fout = fdopen(fd[1], "w");
    if (!fork()) {
        fprintf(fout, "%d\n", getpid());
        fflush(fout);
        while (1) pause();
        _exit(0);
    }   
    if (!fork()) {
        fprintf(fout, "%d\n", getpid());
        fflush(fout);
        while (1) pause();
        _exit(0);
    }
    int pid1, pid2;
    fscanf(fin, "%d%d", &pid1, &pid2);
    unsigned long long val = 1;
    fprintf(fout, "%d\n%d\n%llu\n", pid1, pid2, val); 
    fflush(fout);
    kill(pid1, SIGUSR1);

    while (wait(NULL) > 0) {}
    fclose(fin);
    fclose(fout);
    printf("Done\n");
    return 0;
}




