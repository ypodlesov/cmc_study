#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

const char *SUP = "18446744073709551616";

int
main()
{
    char c;
    long long res = 0;
    int cur = 0, flag = 0;
    while(1) {
        c = getchar();
        if (c == EOF) {
            if (flag == 1) printf("%lld\n", res);
            break;
        } 
        if (c != 'a' && c != '1' && c != '0' && c != ' ' && c != '\n') continue;
        if (!flag && c != 'a' && c != '1' && c != '0') continue;
        if (flag == 2 && c != '\n' && c != ' ') continue;
        if (flag && (c == ' ' || c == '\n')) {
            if (flag == 1) printf("%lld\n", res);
            flag = 0;
            res = 0;
            continue;
        }
        switch (c) {
            case 'a':
                cur = -1;
                break;
            case '1':
                cur = 1;
                break;
            default:
                cur = 0;
                break;
        }
        long long tmp;
        if (__builtin_saddll_overflow(res, res, &tmp) || __builtin_saddll_overflow(tmp, cur, &tmp) || 
                __builtin_saddll_overflow(tmp, res, &tmp)) {
            flag = 2;
            res = 0;
            printf("%s\n", SUP);
            continue;
        }
        res = tmp;
        flag = 1;
    }
    return 0;
}

