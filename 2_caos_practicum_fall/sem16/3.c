#include <stdio.h>
#include <stdlib.h>

unsigned int 
cnt(unsigned int *fat, unsigned int num, unsigned int n) 
{
    unsigned res = 0;
    while (num != n - 1) {
        if (num == n - 2) continue;
        ++res;
        num = fat[num];
    }
    return res;
}

int
main(void)
{
    unsigned int n;
    scanf("%u\n", &n);
    unsigned int *fat = calloc(n, sizeof(*fat));
    for (int i = 2; i < n - 2; ++i) {
        scanf("%u", &fat[i]);
    }
    unsigned int num;
    while (scanf("%u", &num) == 1) {
        printf("%u\n", cnt(fat, num, n));
    }
    free(fat);


}
