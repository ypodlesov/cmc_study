#include <stdio.h>
#include <dlfcn.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#define STDCALL __attribute__((stdcall))

enum
{
    SHIFT = 8
};

void
push_int(char *buf, int val) 
{
    for (int i = 0; i < sizeof(int); ++i) {
        buf[i] = (char) val;
        val >>= SHIFT;
    }
}

void 
push_double(char *buf, double val)
{
    char *tmp = (char *) &val;
    for (int i = 0; i < sizeof(val); ++i) {
        buf[i] = *tmp;
        ++tmp;
    }
}

void 
push_pointer(char *buf, char *val) 
{
    char *tmp = (char *) &val;
    for (int i = 0; i < sizeof(char *); ++i) {
        buf[i] = *tmp;
        ++tmp;
    }
}

struct Args
{
    char buf[64];
};

int
main(int argc, char *argv[]) 
{
    void *handle = dlopen(argv[1], RTLD_LAZY);
    if (handle == NULL) {
        fprintf(stderr, "error! dlopen\n");
        exit(1);
    }
    struct Args args = {};
    char *last = args.buf;
    for (int i = 1; argv[3][i] != '\0'; ++i) {
        if (argv[3][i] == 'i') {
            errno = 0;
            char *eptr = NULL;
            long itmp = strtol(argv[3 + i], &eptr, 10);
            if (*eptr || eptr == argv[3 + i] || (int) itmp != itmp) {
                fprintf(stderr, "error! strtol\n");
                exit(1);
            }
            push_int(last, (int) itmp);
            last += sizeof(int);
        } else if (argv[3][i] == 'd') {
            errno = 0;
            char *eptr = NULL;
            double dtmp = strtod(argv[3 + i], &eptr);
            if (*eptr || eptr == argv[3 + i]) {
                fprintf(stderr, "error! strtol\n");
                exit(1);
            }
            push_double(last, dtmp);
            last += sizeof(double); 
        } else if (argv[3][i] == 's') {
            char *stmp = strdup(argv[3 + i]);
            push_pointer(last, stmp);  
            last += sizeof(char *);
        }
    }
    void *sym = dlsym(handle, argv[2]); 
    if (sym == NULL) {
        fprintf(stderr, "error! dlsym\n");
        exit(1);
    }
    if (argv[3][0] == 'v') {
        ((STDCALL void (*)(struct Args)) sym)(args);
    } else if (argv[3][0] == 'i') {
        printf("%d\n", ((STDCALL int (*)(struct Args)) sym)(args));
    }  else if (argv[3][0] == 'd') {
        printf("%.10g\n", ((STDCALL double (*)(struct Args)) sym)(args));
    } else {
        printf("%s\n", ((STDCALL char* (*)(struct Args)) sym)(args));
    }

    dlclose(handle);
}
