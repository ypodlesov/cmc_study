#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int
main()
{

    double x, y;
    if (scanf("%lf %lf", &x, &y) != 2) {
        fprintf(stderr, "programm: expected 2 floating point numbers for input\n");
        exit(1);
    }
    if (x < 2) {
        printf("0\n");
    } else if (x > 5) {
        printf("0\n");
    } else if (y < 1) {
        printf("0\n");
    } else if (y > 7) {
        printf("0\n");
    } else if (y < x - 2) {
        printf("0\n");
    } else {
        printf("1\n");
    }

    return 0;
}


