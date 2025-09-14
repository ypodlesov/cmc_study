#include <stdio.h>
#include <stdlib.h>


void
print_permutation(int *arr, int cur_size, int n, int *used);

int main(void)
{

    int n;
    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "programm: expected 1 integer number for input\n");
        exit(1);
    }
    if (n < 0) {
        fprintf(stderr, "programm: expected n > 0\n");
        exit(1);
    }
    int *arr = calloc(n + 1, sizeof(n));
    int *used = calloc(n + 1, sizeof(n));
    for (int i = 0; i < n; ++i) {
        used[i] = 0;
    }
    print_permutation(arr, 0, n, used);
    free(arr);
    free(used);

    return 0;
}

void
print_permutation(int *arr, int cur_size, int n, int *used)
{
    if (cur_size == n) {
        for (int i = 0; i < n; ++i) {
            printf("%d", arr[i]);
        }
        printf("\n");
        return;
    }
    for (int i = 1; i <= n; ++i) {
        if (used[i] == 1) {
            continue;
        }
        arr[cur_size] = i;
        used[i] = 1;
        print_permutation(arr, cur_size + 1, n, used);
        used[i] = 0;
    }
}
