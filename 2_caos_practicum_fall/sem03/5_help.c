#include "5.c"

int main()
{

    struct Elem *head = calloc(1, sizeof(struct Elem));
    struct Elem *second= calloc(1, sizeof(struct Elem));
    struct Elem *third = calloc(1, sizeof(struct Elem));
    struct Elem *forth = calloc(1, sizeof(struct Elem));
    head->str = "hello";
    head->next = second;
    second->str = "   -1   ";
    second->next = third;
    third->str = "230";
    third->next = forth;
    forth->str = "good";
    forth->next = NULL;
    head = dup_elem(head);
    struct Elem *it = head;
    while (it != NULL) {
        printf("%s\n", it->str);
        it = it->next;
    }


    return 0;
}
