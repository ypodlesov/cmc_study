#include <stdio.h>
#include <stdlib.h>

int 
bin_pow(int a, int n, int p)
{
    int res = 1;
    while (n) {
        if (n & 1) {
            res = res * a % p;
        }
        a = a * a % p;
        n >>= 1;
    }
    return res;
} 

int
main(void)
{
    int n;
    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "ERROR! Expected 1 decimal number\n");
        exit(1);       
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 1; j < n; ++j) {
            printf("%d ", i * bin_pow(j, n - 2, n) % n);
        }
        printf("\n");
    }

    return 0;
}
