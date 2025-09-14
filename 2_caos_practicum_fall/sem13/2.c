#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>


int
main() 
{
    int fds[2];
    pipe(fds);
    if (fork() > 0) {
        // parent
        close(fds[0]);
        int n;
        while (scanf("%d", &n) == 1) {
            write(fds[1], &n, sizeof(n)); 
        }
        close(fds[1]);
        while (wait(NULL) > 0) {}
        exit(0);
    } 
    if (fork() > 0) {
        // son
        close(fds[0]);
        close(fds[1]);
        wait(NULL);
        exit(0);
    } 
    // grandson
    close(fds[1]);
    long long res = 0;
    int n;
    int flag;
    while ((flag = read(fds[0], &n, sizeof(n))) > 0) {
        res += n;
    }
    close(fds[0]);
    printf("%lld\n", res);
    exit(0);
}
