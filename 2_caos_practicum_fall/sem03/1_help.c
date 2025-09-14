#include <stdio.h>
#include <stdlib.h>

typedef int STYPE;
typedef unsigned int UTYPE;

STYPE bit_reverse(STYPE value);

#include "1.c"

int 
main(void) 
{
		STYPE a;
		scanf("%d", &a);
		a = bit_reverse(a);
		printf("%d\n", a);
}
