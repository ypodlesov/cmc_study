#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int
main() 
{
    int pid;
    if ((pid = fork()) == 0) {
        if ((pid = fork()) == 0) {
            printf("3 ");
            exit(0);
        }
        wait(NULL);
        printf("2 ");
        exit(0);
    }
    wait(NULL);
    printf("1\n");

    return 0;
}
