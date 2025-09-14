#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <string.h>
#include <sys/msg.h>
#include <fcntl.h>


enum
{
    MODE = 0666
};

struct Msg 
{
    long mtype;
    char data[1];
};

const char path[] = "./file";

int
count_addr(int n, int div)
{
    if (div == 0) return 0;
    if (!n) return n;
    if (n > 0) {
        return n % div; 
    }
    return n - (n - div + 1) / div * div;
}

void 
work(int serial, int qid, int count)
{
    struct Msg msg;
    while (1) {
        msgrcv(qid, &msg, sizeof(msg.data), serial, 0);
        if (msg.data[0] == 'e') break;
        int n;
        if (scanf("%d", &n) != 1) {
            msg.mtype = count + 1;
            msgsnd(qid, &msg, sizeof(msg.data), 0); 
            continue;
        }
        printf("%d %d\n", serial - 1, n); fflush(stdout);
        msg.mtype = count_addr(n, count) + 1;
        msgsnd(qid, &msg, sizeof(msg.data), 0);
    }
    msg.mtype = count + 1;
    msgsnd(qid, &msg, sizeof(msg.data), 0); 
}   

int
main(int argc, char *argv[]) 
{
    setvbuf(stdin, NULL, _IONBF, 0);     
    int count = strtol(argv[1], NULL, 10);
    int fd = open(path, O_CREAT | O_TRUNC | O_RDWR, MODE);
    int key = ftok(path, 's');
    int qid = msgget(key, 0666 | IPC_CREAT);
    for (int i = 1; i <= count; ++i) {
        if (!fork()) {
            work(i, qid, count);
            close(fd);
            _exit(0);
        }
    }
    struct Msg msg;
    msg.mtype = 1;
    msg.data[0] = 'r';
    msgsnd(qid, &msg, sizeof(msg.data), 0);
    msgrcv(qid, &msg, sizeof(msg.data), count + 1, 0); 
    msg.data[0] = 'e';
    for (int i = 1; i <= count; ++i) {
        msg.mtype = i;
        msgsnd(qid, &msg, sizeof(msg.data), 0);
        msgrcv(qid, &msg, sizeof(msg.data), count + 1, 0); 
    }
    while (wait(NULL) > 0) {}
    close(fd);
    remove(path);
    msgctl(qid, IPC_RMID, NULL);

    return 0;
}
