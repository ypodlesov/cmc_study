#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>


void 
foo(int *volume, DIR *pdir) 
{
    int res = 0;
    if (pdir == NULL) return;
    struct dirent *dir;
    while ((dir = readdir(pdir)) != NULL) {
        struct stat cur;
        stat(dir->d_name, &cur);
        if (S_ISREG(cur.st_mode)) {
            res += cur.st_size
        }
    }
}

int
main(int argc, char *argv[])
{
    if (argc < 3) {
        return 0;
    }
    char *path = strdup(argv[1]);
    char *eptr = NULL;
    long uid = strtol(argv[2], eptr, 10);
    DIR *pdir = opendir(path);
    int res = 0;
    foo(&res, pdir);
    printf("%d\n", res);



    return 0;
}
