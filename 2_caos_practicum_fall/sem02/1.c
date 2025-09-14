#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void
print_str(char *s)
{
    size_t length = strlen(s);
    for (int i = 0; i < length - 1; ++i) {
        fputc(s[i], stdout);
    }
}

int
main(void)
{

    char *s1, *s2, *s3;
    s1 = calloc(128, sizeof(char));
    s2 = calloc(128, sizeof(char));
    s3 = calloc(128, sizeof(char));
    fgets(s1, 128, stdin);
    fgets(s2, 128, stdin);
    fgets(s3, 128, stdin);
    fputs("[Host:", stdout);
    print_str(s1);
    fputs(",Login:", stdout);
    print_str(s2);
    fputs(",Password:", stdout);
    print_str(s3);
    fputs("]\n", stdout);









    return 0;
}
