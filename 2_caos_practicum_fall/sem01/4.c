#include <stdio.h>
#include <stdlib.h>

int 
cmp(const void *a, const void *b)
{
    int arg1 = *(const int*) a;
    int arg2 = *(const int*) b;
    if (!(arg1 & 1) && arg2 & 1) {
        return -1;
    } else if (!(arg1 & 1) && !(arg2 & 1) && arg2 > arg1) {
        return -1;
    } else if (arg1 & 1 && arg2 & 1 && arg2 < arg1) {
        return -1;
    } else if (!(arg1 & 1) && !(arg2 & 1) && arg2 == arg1) {
        return 0;
    } else if (arg1 & 1 && arg2 & 1 && arg2 == arg1) {
        return 0;
    }
    return 1;
}

void
sort_even_odd(size_t count, int *data)
{
    qsort(data, count, sizeof(int), cmp);
}
