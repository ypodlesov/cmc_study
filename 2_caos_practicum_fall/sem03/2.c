#include <stdio.h>
#include <stdlib.h>

int 
main(int argc, char* argv[]) 
{
    long long positive_sum = 0, negative_sum = 0;
    for (int i = 1; i < argc; ++i) {
        char *eptr = NULL;
        int errno = 0;
        long lval = strtoll(argv[i], &eptr, 10);
        if (errno || *eptr || eptr == argv[i] || (int) lval != lval) {
            fprintf(stderr, "program: expected 32-bit numbers\n");
            exit(1);
        }
        if (lval > 0 && __builtin_add_overflow(positive_sum, lval, &positive_sum)) {
            fprintf(stderr, "program: positive sum overflow\n");
            exit(1);
        } 
        if (lval < 0 && __builtin_add_overflow(negative_sum, lval, &negative_sum)) {
            fprintf(stderr, "program: negative sum overflow\n");
            exit(1);
        }
    }
    printf("%lld\n%lld\n", positive_sum, negative_sum);
    return 0;
}
