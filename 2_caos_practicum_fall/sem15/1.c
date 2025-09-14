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


struct Msg 
{
    long mtype;
    char data[8];
};

unsigned int 
count_addr(unsigned int n, unsigned int mod, int k) 
{
    unsigned int res = 1;
    for (int i = 0; i < k; ++i) {
        res *= n;
        res %= mod;
    }
    return res + 1;
}

char *
push(char *dst, unsigned int src1, unsigned int src2) 
{
    char *tmp = (char *) &src1;
    for (int i = 0; i < sizeof(src1); ++i) {
        dst[i] = tmp[i];
    }
    tmp = (char *) &src2;
    for (int i = 0; i < sizeof(src2); ++i) {
        dst[sizeof(src1) + i] = tmp[i];
    }
    return dst;
}

void
pop(unsigned int *dst, char *src) 
{
    unsigned int *tmp = (unsigned int *) src;
    dst[0] = *tmp;
    tmp = (unsigned int *) &src[4];
    dst[1] = *tmp;
}

void
work(int serial, int qid, int nproc, int maxval) 
{
    struct Msg msg;
    unsigned int valsnder[2];
    while (1) {
        msgrcv(qid, &msg, sizeof(valsnder), serial, 0);
        pop(valsnder, msg.data);
        if (valsnder[0] > maxval) {
            exit(0);
        }
        printf("%d %u %u\n", serial, valsnder[0], valsnder[1]);
        fflush(stdout);
        ++valsnder[0];
        if (valsnder[0] > maxval) {
            msg.mtype = nproc + 1;
            msgsnd(qid, &msg, sizeof(valsnder), 0);
            exit(0);
        }
        msg.mtype = count_addr(valsnder[0], nproc, 4);
        valsnder[1] = serial;
        push(msg.data, valsnder[0], valsnder[1]);
        msgsnd(qid, &msg, sizeof(valsnder), 0);
    }
}

int 
main(int argc, char *argv[]) 
{
    int nproc = strtol(argv[1], NULL, 10);
    int key = strtol(argv[2], NULL, 10);
    //int key = ftok("file", 0);
    //printf("%d\n", key);
    int maxval = strtol(argv[3], NULL, 10);
    if (maxval < 0) _exit(0);
    int qid = msgget(key, 0666 | IPC_CREAT);
    for (int i = 1; i <= nproc; ++i) {
        if (!fork()) {
            work(i, qid, nproc, maxval);    
            exit(0);
        }
    }
    struct Msg msg;
    unsigned int valsnder[2] = {0, 0};
    msg.mtype = 1;
    push(msg.data, valsnder[0], valsnder[1]);
    msgsnd(qid, &msg, sizeof(valsnder), 0);
    msgrcv(qid, &msg, sizeof(valsnder), nproc + 1, 0);
    for (int i = 1; i <= nproc; ++i) {
        msg.mtype = i;
        push(msg.data, maxval + 1, 0);
        msgsnd(qid, &msg, sizeof(valsnder), 0); 
    }
    while (wait(NULL) > 0) {}
    msgctl(qid, IPC_RMID, NULL);
    
    return 0;
}
