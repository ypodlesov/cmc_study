#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int name_check(char *str) {
    int len = strlen(str);
    return (len >= 4 && str[len - 4] == '.' && str[len - 3] == 'e' && str[len - 2] == 'x' && str[len - 1] == 'e');
}


int 
main(int argc, char *argv[]) 
{
    if (argc < 2) {
        fprintf(stderr, "error!\n");
        exit(1);
    }
    DIR *dirp = opendir(argv[1]);
    if (dirp == NULL) {
        fprintf(stderr, "error!\n");
        exit(1);
    }
    char path[PATH_MAX];
    int cnt = 0;
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        strcpy(path, argv[1]);
        strcat(strcat(path, "/"), dp->d_name);
        struct stat file;
        cnt += (stat(path, &file) >= 0 && S_ISREG(file.st_mode) && name_check(path) && !access(path, X_OK));
    }
    closedir(dirp);
    printf("%d\n", cnt);

    return 0;
}
