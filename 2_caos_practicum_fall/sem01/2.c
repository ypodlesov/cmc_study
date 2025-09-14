#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

enum
{
    DIGIT_BEGIN = 1,
    LAST_SYMBOL = 63
};

int
convert(int ch)
{
    if (ch == 0) {
        return '@';
    }
    if (ch >= DIGIT_BEGIN && ch < DIGIT_BEGIN + 10) {
        return ch - DIGIT_BEGIN + '0';
    }
    if (ch >= DIGIT_BEGIN + 10 && ch < DIGIT_BEGIN + 36) {
        return ch - DIGIT_BEGIN - 10 + 'a';
    }
    if (ch >= DIGIT_BEGIN + 36 && ch < LAST_SYMBOL) {
        return ch - DIGIT_BEGIN - 36 + 'A';
    }
    return '#';
}

int
main(void)
{
    int ch = '0';
    while((ch = getchar()) != EOF) {
        if (!isdigit(ch) && !isalpha(ch)) {
            continue;
        }
        if (isdigit(ch)) {
            ch = ch - '0' + DIGIT_BEGIN;
        } else if (islower(ch)) {
            ch = ch - 'a' + DIGIT_BEGIN + 10;
        } else if (isupper(ch)) {
            ch = ch - 'A' + DIGIT_BEGIN + 36;
        }
        ch ^= 1 << 3;
        ch &= ~(1 << 2); // in 46-47 the defect of machine is considered  
        ch = convert(ch); // convert back to punched card mode
        putchar(ch);
    }
    printf("\n");

    return 0;
}
