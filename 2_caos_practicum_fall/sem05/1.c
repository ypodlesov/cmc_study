#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

enum 
{
    KBYTE = 1 << 10
};

int 
main(int argc, char *argv[]) 
{
    off_t res_size = 0;
    for (int i = 1; i < argc; ++i) {
        struct stat *file = calloc(1, sizeof(struct stat));
        if (access(argv[i], F_OK)) {
            continue;
        }
        lstat(argv[i], file);
        if (file->st_size % KBYTE == 0 && S_ISREG(file->st_mode) && !S_ISLNK(file->st_mode) && file->st_nlink == 1) {
            res_size += file->st_size;
        }
    }
    printf("%lld\n", res_size);


    return 0;
}
