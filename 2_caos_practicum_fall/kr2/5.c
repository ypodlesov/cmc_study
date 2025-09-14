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

const char path[] = "./file";


enum
{
    MODE = 0666
};

struct Msg 
{
    long mtype;
    char data[1];
};

void
work(int serial, int l, int qid, int nproc)
{
    struct Msg msg;
    while (1) {
        msgrcv(qid, &msg, sizeof(msg.data), serial, 0);
        if (msg.data[0] == 'S') {
            printf("%d %d\n", serial - 1, --l);
            fflush(stdout);
            msg.mtype = nproc + 1;
            if (!l) {
                msg.data[0] = 'K';
                msgsnd(qid, &msg, sizeof(msg.data), 0);
                _exit(0);
            } else {
                msg.data[0] = 'C';
                msgsnd(qid, &msg, sizeof(msg.data), 0);
            }
        } else if (msg.data[0] == 'E') {
            _exit(0);
        }
    }
}

int
main(int argc, char *argv[])
{
    setvbuf(stdin, NULL, _IONBF, 0);     
    unsigned int nproc = strtoul(argv[1], NULL, 10);
    unsigned int l = strtoul(argv[2], NULL, 10);
    int fd = open(path, O_CREAT | O_TRUNC | O_RDWR, MODE);
    int key = ftok(path, 's');
    int qid = msgget(key, 0666 | IPC_CREAT);
    //printf("%d\n", qid);
    for (int i = 1; i <= nproc; ++i) {
        if (!fork()) {
            work(i, l, qid, nproc);
            close(fd);
            _exit(0);
        }
    }
    struct Msg msg;
    int n = nproc;
    unsigned int num;
    int *killed = calloc(num + 1, sizeof(*killed));
    while (scanf("%d", &num) == 1) {
        if (killed[num % nproc + 1]) continue;
        msg.mtype = num % nproc + 1;
        msg.data[0] = 'S';
        msgsnd(qid, &msg, sizeof(msg.data), 0);
        msgrcv(qid, &msg, sizeof(msg.data), nproc + 1, 0);
        if (msg.data[0] == 'K') {
            --n;
            killed[num % nproc + 1] = 1;
        } 
    }
    printf("%d\n", n);
    fflush(stdout);
    for (int i = 1; i <= nproc; ++i) {
        if (killed[i]) continue;
        msg.mtype = i;
        msg.data[0] = 'E';
        msgsnd(qid, &msg, sizeof(msg.data), 0);
    }
    while (wait(NULL) > 0) {}

    close(fd);
    remove(path);
    msgctl(qid, IPC_RMID, NULL);


    
    return 0;
}
