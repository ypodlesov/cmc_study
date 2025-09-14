#include <stdio.h>

#include "5.c"

int main(int argc, char *argv[]) {
    
    ssize_t size = strtol(argv[1], NULL, 10);
    DIR *dirp = opendir(argv[2]);
    int fd = dirfd(dirp);
    char buf[size];
    size = getcwd2(fd, buf, size);
    printf("%zd %zd %s\n", strlen(buf), size, buf);
    //printf("%d\n", fd);
    

    return 0;
}
