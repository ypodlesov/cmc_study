#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>
#include <limits.h>


int
main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "few args!\n");
        exit(1);
    }
    char path1[PATH_MAX] = {};
    char path2[PATH_MAX] = {};
    strcpy(path1, argv[1]);
    strcpy(path2, argv[2]);
    int len1 = strlen(argv[1]);
    int len2 = strlen(argv[2]);
    path1[len1++] = '/';
    path2[len2++] = '/';
    path1[len1] = '\0';
    path2[len2] = '\0';
    struct dirent *dp1;
    struct stat stb1;
    off_t ans = 0; 
    DIR *dirp1 = opendir(argv[1]);
    if (!dirp1) {
        fprintf(stderr, "error! opendir\n");
        exit(1);
    }
    while ((dp1 = readdir(dirp1)) != NULL) {
        int len_dp1 = strlen(dp1->d_name);
        for (int i = 0; i < len_dp1; ++i) {
            path1[len1 + i] = dp1->d_name[i];
        }
        path1[len1 + len_dp1] = '\0';
        if (lstat(path1, &stb1) || !S_ISREG(stb1.st_mode) || access(path1, W_OK)) {
            continue;
        }
        DIR *dirp2 = opendir(argv[2]);
        struct dirent *dp2;
        while ((dp2 = readdir(dirp2)) != NULL) {
            int len_dp2 = strlen(dp2->d_name);
            for (int i = 0; i < len_dp2; ++i) {
                path2[len2 + i] = dp2->d_name[i];
            }
            path2[len2 + len_dp2] = '\0';
            struct stat stb2;
            if (stat(path2, &stb2)) {
                continue;
            }
            if (!strcmp(dp1->d_name, dp2->d_name) && stb1.st_ino == stb2.st_ino && stb1.st_dev == stb2.st_dev) {
                ans += stb1.st_size;
                break;
            }
        }
        closedir(dirp2);
    }
    closedir(dirp1);
    printf("%llu\n", ans);



    return 0;
}
