#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <sys/wait.h>
#include <pthread.h>

#define POW2(A) ((A) * (A))

int nproc;
int maxval;
int sender;
int rcver;
unsigned int n;

pthread_mutex_t semop;
pthread_cond_t cond_chg;

void 
destroy_all(void) 
{
    pthread_mutex_destroy(&semop);
    pthread_cond_destroy(&cond_chg);
}

void * 
thread_func(void *a) 
{
    int serial = *(int *) a;  
    while (n <= maxval) {
        while (rcver != serial) {
            pthread_cond_wait(&cond_chg, &semop);
            if (n > maxval) {
                pthread_mutex_unlock(&semop);
                pthread_cond_broadcast(&cond_chg);
                pthread_exit(0);
            }
        }
        printf("%d %d %d\n", serial, n, sender); 
        fflush(stdout);
        ++n;
        sender = rcver;
        rcver = POW2(POW2(n % nproc) % nproc) % nproc + 1;
        pthread_mutex_unlock(&semop);
        pthread_cond_broadcast(&cond_chg);
        if (n > maxval) pthread_exit(0);
    }
    return NULL;
}


int
main(int argc, char *argv[]) {

    nproc = strtol(argv[1], NULL, 10);
    maxval = strtol(argv[3], NULL, 10);
    if (maxval <= 0) return 0;
    pthread_t th[nproc + 1];
    pthread_mutex_init(&semop, NULL);
    pthread_cond_init(&cond_chg, NULL);
    int tmp[nproc + 1];
    for (int i = 1; i <= nproc; ++i) {
        tmp[i] = i;
        if (pthread_create(&th[i], NULL, &thread_func, tmp + i)) {
            fprintf(stderr, "cannot create a thread!\n");
            destroy_all();
            exit(0);
        } 
    }
    sender = 0;
    rcver = 1;
    n = 0;
    if (pthread_cond_broadcast(&cond_chg)) {
        fprintf(stderr, "cannot send broadcast\n");
        destroy_all();
        exit(0);
    }
    for (int i = 1; i <= nproc; ++i) {
        if (pthread_join(th[i], NULL)) {
            destroy_all();
            fprintf(stderr, "cannot join a thread\n");
            exit(0);
        }
    }
    destroy_all();

    return 0;
}
