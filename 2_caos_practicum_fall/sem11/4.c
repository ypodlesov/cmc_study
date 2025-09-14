#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <errno.h>

int
main() 
{   
    int n;
    scanf("%d", &n);
    printf("1");
    if (n == 1) printf("\n");
    else printf(" ");
    fflush(stdout);
    for (int i = 2; i <= n; ++i) {
        int pid;
        if (!(pid = fork())) {
            if (i == n) printf("%d\n", i);
            else printf("%d ", i);
            fflush(stdout);
        } else if (pid > 0) {
            wait(NULL);
            exit(0);
        }
    }
    wait(NULL);
    return 0;
}
