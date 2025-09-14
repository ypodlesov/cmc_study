#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

int
main(int argc, char *argv[])
{
    long long res = 0;
    for (int i = 1; i < argc; ++i) {
        errno = 0;
        char *eptr = NULL;
        long cur = strtol(argv[i], &eptr, 10);
        if (errno || eptr == argv[i] || (int) cur != cur) continue;
        int len = strlen(eptr);
        if (eptr[0] == '+') {
            if (len > 1) continue;
            res += cur;
        } else if (eptr[0] == '-') {
            if (len > 1) continue;
            res -= cur;
        } else if (eptr[0] == 'k') {
            if (len > 2) continue;
            int tmp;
            if (__builtin_smul_overflow((int) cur, 1000, &tmp)) continue;
            if (eptr[1] == '+') {
                res += tmp;
            } else if (eptr[1] == '-') {
                res -= tmp;
            }
        }
    }
    printf("%lld\n", res);

    return 0;
}
