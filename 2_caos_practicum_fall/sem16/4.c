#include <stdio.h>
#include <stdlib.h>


#define MAX(A, B) ((A) > (B) ? (A) : (B))

unsigned int
mpow(unsigned int n) {
    long long res = 0, cur = 1;
    while (cur <= n) {
        if (n % cur == 0) {
            res = cur;
        }
        cur <<= 1;
    }
    return (unsigned int) res;
}

int
main(void)
{
    unsigned int tmp;
    unsigned int cur_align = 0;
    unsigned int align = 0;
    unsigned int cur = 0;
    while ((scanf("%u", &tmp)) == 1) {
        if (!tmp) continue;
        cur_align = mpow(tmp);
        cur = (cur + cur_align - 1) / cur_align * cur_align;
        cur += tmp;
        align = MAX(align, cur_align);
    }
    if (align) cur = (cur + align - 1) / align * align;
    printf("%u %u\n", cur == 0 ? 1 : cur , align == 0 ? 1 : align);


    return 0;
}
