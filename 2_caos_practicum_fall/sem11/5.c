#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <stdlib.h>

int
main()
{
    int pid = fork();
    if (pid < 0) {
        printf("-1\n");
        exit(0);
    } else if (pid == 0) {
        while (1) {
            int n;
            if (scanf("%d", &n) != 1) exit(0);
            pid = fork();
            if (pid < 0) {
                exit(1);
            } else if (pid > 0) {
                int status;
                wait(&status);
                if (WEXITSTATUS(status)) {
                    exit(1);
                } else {
                    printf("%d\n", n);
                    exit(0);
                }
            }
        }
    } else {
        int status;
        wait(&status);
        if (WEXITSTATUS(status)) {
            printf("-1\n");
        }
        return 0;
    }
    return 0;
}
