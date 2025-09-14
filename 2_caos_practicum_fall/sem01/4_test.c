// #include <stdio.h>
// #include <stdlib.h>
#include "4.c"

int
main(void)
{

    int n;
    scanf("%d", &n);
    int *a = calloc(n, sizeof(n));
    for (int i = 0; i < n; ++i) {
        scanf("%d", &a[i]);
    }
    sort_even_odd(n, a);
    for (int i = 0; i < n; ++i) {
        printf("%d ", a[i]);
    }
    free(a);
    printf("\n");

    return 0;
}



// 10
// 5 3 6 7 8 1 2 10 9 4
