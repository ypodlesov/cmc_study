#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

unsigned long long m;

void
work(int serial, FILE *fin, FILE *fout)
{
    while (1) {
        unsigned long long val;
        if (fscanf(fin, "%llu", &val) == EOF || val >= m) {
            fflush(stdout);
            fclose(fin);
            fclose(fout);
            _exit(0);
        }
        printf("%d %llu\n", serial, val);
        fflush(stdout);
        if (fprintf(fout, "%llu\n", ++val) == -1 || val >= m) {
            fclose(fin);
            fclose(fout);
            _exit(0);
        }
        fflush(fout);
    }
}

int
main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    int fd1[2];
    int fd2[2];
    pipe(fd1);
    pipe(fd2);
    FILE *fin1 = fdopen(fd1[0], "r");
    FILE *fout1 = fdopen(fd1[1], "w");
    FILE *fin2 = fdopen(fd2[0], "r");
    FILE *fout2 = fdopen(fd2[1], "w");
    m = strtoull(argv[1], NULL, 10);
    if (!fork()) {
        fclose(fin2);
        fclose(fout1);
        work(1, fin1, fout2);
        _exit(0);
    } 
    if (!fork()) {
        fclose(fin1);
        fclose(fout2);
        work(2, fin2, fout1);
        _exit(0);
    }
    fclose(fin1);
    fclose(fin2);
    fclose(fout2);
    unsigned long long val = 1;
    fprintf(fout1, "%llu\n", val);
    fflush(fout1);
    fclose(fout1);
    
    while (wait(NULL) > 0) {}
    printf("Done\n");
    fflush(stdout);
    return 0;
}

