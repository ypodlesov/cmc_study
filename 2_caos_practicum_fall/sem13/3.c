#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>

#define SUCCESS(A) (WIFEXITED(A) && !WEXITSTATUS(A))

enum
{
    MODE = 0666
};

int
main(int argc, char *argv[]) 
{
    if (argc < 6) {
        fprintf(stderr, "few arguments!\n");
        exit(1);
    }
    int fds[2] = {};
    pipe(fds);
    if (!fork()) {
        close(fds[1]);
        int file2_des = open(argv[5], O_CREAT | O_WRONLY | O_APPEND, MODE);      
        dup2(file2_des, STDOUT_FILENO);
        close(file2_des);
        dup2(fds[0], STDIN_FILENO);
        close(fds[0]);
        execlp(argv[3], argv[3], NULL);
        _exit(1);
    } 
    if (!fork()) {
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        close(fds[1]);
        if (!fork()) {
            int file1_des = open(argv[4], O_RDONLY);
            dup2(file1_des, STDIN_FILENO);
            close(file1_des);
            execlp(argv[1], argv[1], NULL);
            _exit(1);
        }
        int status;
        wait(&status);
        if (!SUCCESS(status)) _exit(1); 
        if (!fork()) {
            execlp(argv[2], argv[2], NULL);
            _exit(1);
        }
        while (wait(NULL) > 0) {}
        _exit(0);
    }
    close(fds[0]);
    close(fds[1]);
    while (wait(NULL) > 0) {}
    return 0;
}

