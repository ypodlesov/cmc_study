#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>

int 
main(int argc, char* argv[]) 
{

    if (argc < 2) {
        fprintf(stderr, "program: need at least 2 arguments\n");
        exit(1);
    }
    char *end_ptr = NULL;
    errno = 0;
    double rate = strtod(argv[1], &end_ptr);
    if (errno || errno == ERANGE || *end_ptr || end_ptr == argv[1]) {
        fprintf(stderr, "program: incorrect input of floating point number\n");
        exit(1);
    }
    for (int i = 2; argv[i] != NULL; ++i) {
        double delta = strtod(argv[i], &end_ptr);
        if (errno || errno == ERANGE || *end_ptr || end_ptr == argv[i]) {
            fprintf(stderr, "program: incorrect input of floating point number\n");
            exit(1);
        }
        rate = round((rate + rate * delta * 0.01) * 1e4) * 1e-4;
    }
    printf("%.4lf\n", rate);

    return 0;
}


















