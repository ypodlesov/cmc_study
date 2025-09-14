#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>

int
main(int argc, char *argv[])
{
    int fds[2];
    pipe(fds);
    for (int i = 0; i < 3; ++i) {
        if (fork() > 0) {
            wait(NULL);
            time_t t;
            read(fds[0], &t, sizeof(t));
            struct tm tmp;
            localtime_r(&t, &tmp);
            if (i == 0) {
                printf("Y:%d\n", tmp.tm_year + 1900); 
                fflush(stdout);
                close(fds[0]);
                close(fds[1]);
                _exit(0);
            } else if (i == 1) {
                printf("M:%02d\n", tmp.tm_mon + 1); 
                fflush(stdout);
                close(fds[0]);
                close(fds[1]);
                _exit(0);
            } else {
                printf("D:%02d\n", tmp.tm_mday); 
                fflush(stdout);
                close(fds[0]);
                close(fds[1]);
                _exit(0);
            }

        }
    }
    time_t t = time(NULL);
    for (int j = 0; j < 3; ++j) {
        write(fds[1], &t, sizeof(t));
    }
    close(fds[1]);
    close(fds[0]);
    _exit(0);

    return 0;
}
