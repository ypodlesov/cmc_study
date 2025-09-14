#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>

#include "5.c"
//#include "YEGOR.c"

int copy_file(const char *srcpath, const char *dstpath);

int 
main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "few args!\n");
        exit(1);
    }
    printf("%d\n", copy_file(argv[1], argv[2]));

    return 0;
}
