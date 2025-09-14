#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <limits.h>
#include <fcntl.h>
#include <sys/wait.h>


#define SUCCESS(A) (WIFEXITED(A) && !(WEXITSTATUS(A)))

int
main(int argc, char *argv[]) 
{
    setvbuf(stdin, NULL, _IONBF, 0);
    int num;
    int flag = 1;
    scanf("%d", &num);
    while (1) {
        if (flag) printf("%d %d\n", getpid(), num);
        fflush(stdout);
        flag = 0;
        int cur_depth = 0;
        int fd[2];
        pipe(fd);
        while (1) {
            if (!fork()) {
                close(fd[0]);
                if (++cur_depth == num) {
                    if (scanf("%d", &num) == 1) {
                        printf("%d %d\n", getpid(), num);
                        fflush(stdout);
                        write(fd[1], &num, sizeof(num));
                        close(fd[1]);
                        _exit(0);
                    } else {
                        close(fd[1]);
                        _exit(1);
                    }
                }
            } else {
                close(fd[1]);
                int status;
                wait(&status);
                if (!cur_depth) {
                    read(fd[0], &num, sizeof(num));
                    if (!SUCCESS(status)) {
                        close(fd[0]);
                        break;
                    } else {
                        close(fd[0]);
                        continue;
                    }
                }
                close(fd[0]);
                _exit(0);
            }
        }
    }
    while (wait(NULL) > 0) {}
    

    return 0;
}
