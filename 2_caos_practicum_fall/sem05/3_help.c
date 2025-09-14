#include "3.c"
#include <stdlib.h>

int 
main() 
{
    char *str = calloc(10, sizeof(char));
    scanf("%s", str);
    printf("%d\n", parse_rwx_permissions(str));

    return 0;
}
