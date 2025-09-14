#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <limits.h>
#include <fcntl.h>

ssize_t 
foo(int fd, char *buf, size_t size) 
{
    if (fchdir(fd) == -1) {
        printf("not ok\n");
    }
    printf("%s\n", getcwd(NULL, 0));
}

int
main(int argc, char *argv[]) 
{

    

    return 0;
}







