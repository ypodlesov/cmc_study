#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <limits.h>

int 
main(int argc, char *argv[]) 
{
    if (argc < 2) {
        fprintf(stderr, "few argumnets!\n");
        exit(1);
    }
    char *str = argv[1];
    int fd = open(str, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "cannot open file\n");
        exit(1);
    }
    long long min_num;
    off_t file_end = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    off_t min_pos = 0;
    if (file_end == 0) {
        return 0;
    }
    if (read(fd, &min_num, sizeof(min_num)) < sizeof(min_num)) {
        fprintf(stderr, "cannot read\n");
        exit(1);
    }
    off_t cur_pos = lseek(fd, 0, SEEK_CUR);
    while (cur_pos < file_end) {
        long long tmp;
        if (read(fd, &tmp, sizeof(tmp)) < sizeof(tmp)) {
            fprintf(stderr, "cannot read!\n");
            exit(1);
        }
        if (tmp < min_num) {
            min_num = tmp;
            min_pos = cur_pos;
        }
        cur_pos += sizeof(tmp);
    }
    lseek(fd, min_pos, SEEK_SET);
    if (min_num > LLONG_MIN) {
        min_num *= -1;
    }   
    if (write(fd, &min_num, sizeof(min_num)) < sizeof(min_num)) {
        fprintf(stderr, "cannot write data\n");
        exit(1);
    }
    close(fd);

    return 0;
}






















