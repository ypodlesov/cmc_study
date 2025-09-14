#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <sys/wait.h>

#define min(A, B) ((A) < (B) ? (A) : (B))
#define SUCCESS(A) (WIFEXITED(A) && !WEXITSTATUS(A))

int
main(int argc, char *argv[])
{
    int n = strtol(argv[1], NULL, 10);
    for (int i = 2; i < min(n + 2, argc); ++i) {
        if (!fork()) {
            FILE *file = fopen(argv[i], "r");
            char prog[PATH_MAX];
            fscanf(file, "%s", prog);
            fclose(file);
            execlp(prog, prog, NULL);
            _exit(1);
        } 
    }
    int status;
    int cnt = 0;
    while (wait(&status) > 0) {
        if (SUCCESS(status)) ++cnt;
    }
    for (int i = n + 2; i < argc; ++i) {
        if (!fork()) {
            FILE *file = fopen(argv[i], "r");
            char prog[PATH_MAX];
            fscanf(file, "%s", prog);
            fclose(file);
            execlp(prog, prog, NULL);
            _exit(1);
        }
        wait(&status);
        if (SUCCESS(status)) ++cnt;
    }
    printf("%d\n", cnt);

    return 0;
}


