#include <stdio.h>
#include <stdlib.h>

#include "2.c"

int
main() {
    
    char *buf = calloc(1024, sizeof(*buf));
    scanf("%s", buf);
    normalize_path(buf);
    printf("%s\n", buf);

    return 0;
}
