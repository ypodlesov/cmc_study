#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int mode = 0;

void
sum_handler(int sig)
{
    mode = 0;
}

void
prod_handler(int sig) 
{
    mode = 1;
}

int
main(void)
{
    struct sigaction sum_act = {};
    sum_act.sa_handler = sum_handler;
    sum_act.sa_flags = SA_RESTART;
    struct sigaction prod_act = {};
    prod_act.sa_handler = prod_handler;
    prod_act.sa_flags = SA_RESTART;
    sigaction(SIGINT, &sum_act, NULL);
    sigaction(SIGQUIT, &prod_act, NULL);
    printf("%d\n", getpid());
    fflush(stdout);
    unsigned res = 0;
    unsigned cur = 0;
    while (scanf("%d", &cur) == 1) {
        if (!mode) res += cur;
        else res *= cur;
        printf("%d\n", res);
        fflush(stdout);
    }
}


