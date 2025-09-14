#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <dirent.h>
#include <limits.h>

#define min(A, B) ((A) < (B) ? (A) : (B))
#define max(A, B) ((A) > (B) ? (A) : (B))

const char *CUR = ".";
const char *PREV = "..";
const char SLASH = '/';
const char NUL = '\0';

ssize_t 
getcwd2(int fd, char *buf, size_t size)
{
    if (size < 0) {
        return -1;
    }
    DIR *init_dirp = opendir(CUR);
    if (init_dirp == NULL) {
        return -1;
    }
    int init_fd = dirfd(init_dirp);
    if (init_fd < 0) {
        return -1;
    }
    //printf("%p\t%d\n", init_dirp, init_fd);
    if (fchdir(fd)) {
        return -1;
    }

    ssize_t path_len = 0;
    char path[PATH_MAX] = {};
    struct stat p_stb;
    struct stat ch_stb;
    lstat(PREV, &p_stb);
    lstat(CUR, &ch_stb);
    DIR *p_dirp = opendir(PREV);
    if (p_dirp == NULL) {
        return -1;
    }
    if (chdir(PREV)) {
        return -1;
    }
    struct dirent *dp;
    while ((dp = readdir(p_dirp)) != NULL) {
        if (!strcmp(dp->d_name, CUR) || !strcmp(dp->d_name, PREV)) {
            continue;
        }
        struct stat check_stb;
        if (lstat(dp->d_name, &check_stb)) {
            return -1;
        }
        if (check_stb.st_ino == ch_stb.st_ino && check_stb.st_dev == ch_stb.st_dev) {
            size_t len = strlen(dp->d_name);
            path[path_len++] = SLASH;
            for (int i = 0; i < len; ++i) {
                path[path_len + i] = dp->d_name[len - i - 1];
            }
            path_len += len;
            closedir(p_dirp);
            
            if (lstat(CUR, &ch_stb)) {
                return -1;
            }
            if (lstat(PREV, &p_stb)) {
                return -1;
            }
            if ((p_dirp = opendir(PREV)) == NULL) {
                return -1;
            }
            if (chdir(PREV)) {
                return -1;
            }
        }
    }
    closedir(p_dirp);
    path[path_len++] = SLASH;
    
    if (fchdir(init_fd)) {
        return -1;
    }
    closedir(init_dirp);
    if (buf == NULL) {
        return path_len;
    }
    for (int i = 0; i < path_len / 2; ++i) {
        char tmp = path[i];
        path[i] = path[path_len - i - 1];
        path[path_len - i - 1] = tmp;
    }
    path[max(1, path_len - 1)] = NUL;
    if (size) {
        strncpy(buf, path, size - 1);
        buf[size - 1] = NUL;
    }
    return max(1, path_len - 1);
}







