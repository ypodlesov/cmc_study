#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

enum
{
    RADIX = 17
};

int 
cmp(const void *a, const void *b) 
{
    int aa = *(long long *) a;
    int bb = *(long long *) b;
    if (aa > bb) {
        return -1;
    } else if (aa == bb) {
        return 0;
    }
    return 1;
}

int
main(int argc, char *argv[])
{
    if (argc < 2) {
        return 0;
    }
    long long *arr = calloc(argc - 1, sizeof(*arr));
    for (int i = 1; i < argc; ++i) {
        errno = 0;
        char *eptr = NULL;
        long long tmp = strtoll(argv[i], &eptr, RADIX);
        if (errno || *eptr || eptr == argv[i]) {
            fprintf(stderr, "error!\n");
            exit(1);
        }
        arr[i-1] = tmp;
    }
    qsort(arr, argc - 1, sizeof(*arr), cmp);
    for (int i = 0; i < argc - 1; ++i) {
        printf("%lld\n", arr[i]);
    }
    free(arr);
    return 0;
}
