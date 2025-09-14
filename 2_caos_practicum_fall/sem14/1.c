#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void 
handler(int sig)
{
    static int cnt = 0;
    if (cnt == 5) exit(0);
    printf("%d\n", cnt++);
    fflush(stdout);
}

int
main(void)
{
    struct sigaction act = {};
    act.sa_handler = handler;
    act.sa_flags = SA_RESTART;
    sigaction(SIGHUP, &act, NULL);
    printf("%d\n", getpid());
    fflush(stdout);
    while (1) {
        pause();
    }
    return 0;
}
