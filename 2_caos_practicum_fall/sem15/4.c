#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <string.h>
#include <sys/msg.h>

enum
{
    MODE = 0666,
    BUFSZ = 32
};  


struct Msg 
{
    long long mtype;
    char data[BUFSZ];
};

void
work(long long serial, int qid, long long nproc, long long maxval) 
{
    struct Msg msg;
    long long x1, x2, x3;
    while (1) {
        msgrcv(qid, &msg, BUFSZ, serial, 0);
        sscanf(msg.data, "%lld%lld", &x1, &x2);
        if (x1 + x2 < 0) {
            exit(0);
        }
        if (__builtin_saddll_overflow(x1, x2, &x3) || x3 > maxval) {
            printf("%lld %lld\n", serial - 1, x3);
            fflush(stdout);
            msg.mtype = nproc + 1;
            msgsnd(qid, &msg, BUFSZ, 0);
            continue;
        }
        printf("%lld %lld\n", serial - 1, x3);
        fflush(stdout);
        if (x3 > maxval || -x3 > maxval) {
            msg.mtype = nproc + 1;
            msgsnd(qid, &msg, BUFSZ, 0);
            exit(0);
        }
        msg.mtype = x3 % nproc + 1;
        sprintf(msg.data, "%lld\n%lld\n", x2, x3);
        msgsnd(qid, &msg, BUFSZ, 0);
    }
}

int 
main(int argc, char *argv[]) 
{
    int key = strtol(argv[1], NULL, 10);
    long long nproc = strtoull(argv[2], NULL, 10);
    //int key = ftok("file", 0);
    //printf("%d\n", key);
    long long x1 = strtoll(argv[3], NULL, 10);
    long long x2 = strtoll(argv[4], NULL, 10);
    long long maxval = strtoll(argv[5], NULL, 10);
    if (maxval < 0) _exit(0);
    if (nproc <= 0) _exit(0);
    int qid = msgget(key, MODE | IPC_CREAT);
    struct Msg msg;
    for (unsigned long long i = 1; i <= nproc; ++i) {
        int pid = fork();
        if (pid < 0) {
            for (long long j = 1; j < i; ++j) {
                msg.mtype = j;
                sprintf(msg.data, "-1\n-1\n");
                msgsnd(qid, &msg, BUFSZ, 0); 
            }
            while (wait(NULL) > 0) {}
            msgctl(qid, IPC_RMID, NULL);
            _exit(1); 
        } else if (!pid) {
            work(i, qid, nproc, maxval);    
            exit(0);
        } 
    }
    msg.mtype = 1;
    sprintf(msg.data, "%lld\n%lld\n", x1, x2);
    msgsnd(qid, &msg, BUFSZ, 0);
    msgrcv(qid, &msg, BUFSZ, nproc + 1, 0);
    for (long long i = 1; i <= nproc; ++i) {
        msg.mtype = i;
        sprintf(msg.data, "-1\n-1\n");
        msgsnd(qid, &msg, BUFSZ, 0); 
    }
    while (wait(NULL) > 0) {}
    msgctl(qid, IPC_RMID, NULL);
    
    return 0;
}
