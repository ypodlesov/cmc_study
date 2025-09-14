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
    for (int i = 1; i <= 3; ++i) {
        int pid;
        if (!(pid = fork())) {
            char str[8];
            read(0, str, sizeof(str));
            long n = strtol(str, NULL, 10);
            printf("%d %d\n", i, (int) (n * n));
            exit(0);
        }
    }
    for (int i = 0; i < 3; ++i) wait(NULL);

    return 0;
}
