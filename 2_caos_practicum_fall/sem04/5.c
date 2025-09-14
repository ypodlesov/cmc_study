#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

enum 
{
    MODE = 0666,
    CHAR_BIT = 8
};

int 
main(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr, "not enough args. error!\n");
        exit(1);
    }
    char *FILE1 = argv[1];
    char *FILE2 = argv[2];
    int fd1 = open(FILE1, O_RDONLY);
    if (fd1 < 0) {
        fprintf(stderr, "cannot open FILE1\n");
        exit(1);
    }
    int fd2 = open(FILE2, O_WRONLY | O_TRUNC | O_CREAT, MODE);
    if (fd2 < 0) {
        fprintf(stderr, "cannot open FILE2\n");
        exit(1);
    }
    errno = 0;
    char *endptr = NULL;
    long lval = strtol(argv[3], &endptr, 10);
    if (errno || *endptr || endptr == argv[3] || (int) lval != lval) {
        fprintf(stderr, "strtol: error!\n");
        exit(1);
    }
    int MOD = lval;
    if (!MOD) {
        fprintf(stderr, "incorrect MOD\n");
        exit(1);
    }
    unsigned long long cur_elem = 0, cur_sum = 0;
    off_t file_end = lseek(fd1, 0, SEEK_END);
    off_t cur_pos = lseek(fd1, 0, SEEK_SET);
    lseek(fd2, 0, SEEK_SET);
    while (cur_pos < file_end) {
        unsigned char ch;
        if (read(fd1, &ch, sizeof(ch)) < sizeof(ch)) {
            fprintf(stderr, "cannot read from FILE1\n");
            exit(1);
        }
        for (int i = 0; i < CHAR_BIT; ++i) {
            cur_elem = (cur_elem + 1) % MOD;
            cur_sum = (cur_sum + cur_elem * cur_elem % MOD) % MOD;
            if ((ch >> i) & 1) {
                unsigned int res = cur_sum;
                printf("%d\n", res);
                if (write(fd2, &res, sizeof(res)) < sizeof(res)) {
                    fprintf(stderr, "cannot write in FILE2\n");
                    exit(1);
                }
            }
        }
        cur_pos += sizeof(ch);
    }
    


    

    return 0;
}































