#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>

enum
{   
    MODE = 0700, 
    FILE_SIZE = 1 << 20,
    MAX_LEN = 10,
    MIN_LEN = 5
};

char *
gen_name(int size) 
{
    srand(time(NULL));
    char *res = calloc(size, sizeof(*res));    
    for (int i = 0; i < size; ++i) {
        res[i] = 'a' + rand() % 26;
    }
    return res;
}

char *
base_name(void)
{
    if (getenv("XDG_RUNTIME_DIR")) {
        return getenv("XDG_RUNTIME_DIR");
    }
    if (getenv("TMPDIR")) {
        return getenv("TMPDIR");
    }
    return strdup("/tmp");
}

int
main(int argc, char *argv[])
{
    if (argc < 2) {
        return 0;
    }
    char content[FILE_SIZE] = "#! /bin/python3\n";
    strcat(content, "import sys\n");
    strcat(content, "from os import remove\n");
    strcat(content, "from sys import argv\n");
    strcat(content, "sys.set_int_max_str_digits(1000000)\n");
    strcat(content, "print(");
    for (int i = 1; i < argc; ++i) {
        strcat(content, argv[i]);
        strcat(content, i == argc - 1 ? ")\n" : " * ");
    }
    strcat(content, "remove(argv[0])\n");
    char path[PATH_MAX];
    strcpy(path, base_name());
    strcat(path, "/");
    char *name = gen_name(MIN_LEN + rand() % MAX_LEN);
    strcat(path, name);
    free(name);
    int fd = open(path, O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC, MODE);
    if (fd < 0) exit(1);
    FILE *file = fdopen(fd, "w");
    fprintf(file, "%s\n", content);
    fclose(file);
    execlp(path, path, NULL);
    exit(1);
}
