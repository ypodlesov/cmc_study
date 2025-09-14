#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

struct Elem
{
    struct Elem *next;
    char *str;
};

struct Elem *
dup_elem(struct Elem *head) 
{
    if (head == NULL) {
        return head;
    }
    struct Elem *cur = head;
    struct Elem *prev = NULL;
    while (cur != NULL) {
        char *end_ptr = NULL;
        errno = 0;
        long long x = strtol(cur->str, &end_ptr, 10);
        if (errno == 0 && *end_ptr == 0 && (end_ptr != cur->str) && (int)x == x && x != (1U << 31) - 1) {
            struct Elem *new_elem = calloc(1, sizeof(*cur));
            if (head == cur) {
                head = new_elem;
                head->next = cur;
            } else {
                new_elem->next = cur;
                prev->next = new_elem;
            }
            new_elem->str = calloc(strlen(cur->str) + 1, sizeof(cur->str[0]));
            if (sprintf(new_elem->str, "%d", (int)x + 1) < 0) {
                fprintf(stderr, "program: error!\n");
                exit(1);
            }
        }
        prev = cur;
        cur = cur->next;
    }
    return head;
}





