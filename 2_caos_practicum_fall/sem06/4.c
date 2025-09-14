#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/dir.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>

enum
{
    MAX_DEPTH = 4
};

void
solve(
        const int base_len, 
        int last_len, 
        char *path, 
        DIR *dir, 
        const int depth, 
        const int Z
        ) 
{
    if (dir == NULL || depth > MAX_DEPTH) {
        return;
    }
    path[last_len] = '/';
    path[++last_len] = '\0';
    struct dirent *dp;
    while ((dp = readdir(dir)) != NULL) {
        if (strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0) {
            continue;
        }
        int cur_len = strlen(dp->d_name);
        for (int i = 0; i < cur_len; ++i) {
            path[i + last_len] = dp->d_name[i];
        }
        path[cur_len + last_len] = '\0';
        struct stat stb;
        if (lstat(path, &stb)) {
            continue;
        }
        if (S_ISREG(stb.st_mode) && !access(path, R_OK) && stb.st_size <= Z) {
            printf("%s\n", &(path[base_len + 1]));
        } else if (S_ISDIR(stb.st_mode)) {
            DIR *cur = opendir(path);
            solve(base_len, cur_len + last_len, path, cur, depth + 1, Z);
        }
    }
    closedir(dir);
}

int 
main(int argc, char *argv[])
{
    if (argc < 3) {
        return 0;
    }
    long tmp;
    char *eptr = NULL;
    errno = 0;
    tmp = strtol(argv[2], &eptr, 10);
    if (errno || *eptr || argv[2] == eptr || (int) tmp != tmp) {
        fprintf(stderr, "strtoll: error!\n");
        exit(1);
    }
    DIR *D = opendir(argv[1]);
    int Z = tmp;
    int len = strlen(argv[1]);
    char path[PATH_MAX] = {};
    strcat(path, argv[1]);
    solve(len, len, path, D, 1, Z);
}
