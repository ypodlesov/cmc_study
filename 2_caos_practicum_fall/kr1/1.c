#include <stdio.h>
#include <stdlib.h>

int 
cmp(const void *a, const void *b) 
{
    long long aa = *(long long *) a;
    long long bb = *(long long *) b;
    if (aa > bb) return 1;
    if (aa == bb) return 0;
    return -1;
}

int
main()
{
    long long num;
    long long *arr = calloc(1, sizeof(*arr));
    int cap = 1;
    int size = 0;
    while(scanf("%lld", &num) == 1) {
        if (cap == size) {
            cap *= 2;
            arr = realloc(arr, cap * sizeof(*arr));
        }
        arr[size++] = num;
    }
    qsort(arr, size, sizeof(*arr), cmp);
    long long res = 0;
    for (int i = 0; i < size / 2; ++i) {
        res = arr[i] + arr[size - i - 1];
    }
    res = size & 1 ? res + arr[size / 2] : res;
    printf("%lld\n", res);


    free(arr);
    return 0;
}
