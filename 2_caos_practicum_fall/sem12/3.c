#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>


int
main(int argc, char *argv[]) 
{
    int ans = 0;
    int status;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == 's') {
            while (wait(&status) > 0) {
                ans += (WIFEXITED(status) && !WEXITSTATUS(status));
            }
        } 
        if (!fork()) {
            execlp(&argv[i][1], &argv[i][1], (char *) NULL);
            _exit(1);
        }
    }
    while (wait(&status) > 0) {
        ans += (WIFEXITED(status) && !WEXITSTATUS(status));
    }
    printf("%d\n", ans);
    return 0;
}
