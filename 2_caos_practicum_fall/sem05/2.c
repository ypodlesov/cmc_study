#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

const char *MODE_PATTERN = "rwxrwxrwx";

int 
main(int argc, char *argv[]) 
{
    for (int i = 1; i < argc; ++i) {
        char *endptr = NULL;
        errno = 0;
        long lval = strtol(argv[i], &endptr, 8);
        if (errno || *endptr || argv[i] == endptr || (int) lval != lval) {
            fprintf(stderr, "cannot convert\n");
            exit(1);
        }
        int cur_mode = lval;
        for (int i = 8; i >= 0; --i) {
            if (cur_mode & 1 << i) {
                printf("%c", MODE_PATTERN[8 - i]);
            } else {
                printf("-");
            }
        }
        printf("\n");
    }
    return 0;
}
