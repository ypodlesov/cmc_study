#include <stdio.h>
#include <string.h>

const char *MODE_STR = "rwxrwxrwx";

int
parse_rwx_permissions(const char *str) 
{
    if (str == NULL) {
        return -1;
    }
    int res = 0;
    int len = strlen(MODE_STR);
    if (len != strlen(str)) {
        return -1;
    }
    for (int i = 0; i < len; ++i) {
        res <<= 1;
        res += (str[i] == MODE_STR[i]); 
        if (str[i] != MODE_STR[i] && str[i] != '-') {
            return -1;
        }
    }
    return res;
}
