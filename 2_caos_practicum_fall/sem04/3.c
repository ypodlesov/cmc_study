#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Node 
{
    int32_t key;
    int32_t left_idx;
    int32_t right_idx;
} Node;

void 
traverse_tree(int fd) 
{
    Node *cur = calloc(1, sizeof(Node));
    if (cur == NULL) {
        fprintf(stderr, "cannot allocate memory\n");
        exit(1);
    }
    if (read(fd, cur, sizeof(Node)) < sizeof(Node)) {
        fprintf(stderr, "cannot read!\n");
        exit(1);
    }
    if (cur->right_idx > 0) {
        lseek(fd, cur->right_idx * sizeof(Node), SEEK_SET);
        traverse_tree(fd);
    } 
    printf("%d ", cur->key);
    if (cur->left_idx > 0) { 
        lseek(fd, cur->left_idx * sizeof(Node), SEEK_SET);
        traverse_tree(fd);
    } 
}

int 
main(int argc, char *argv[]) 
{
    if (argc < 2) {
        fprintf(stderr, "few args!\n");
        exit(1);
    }
    char *path = argv[1];
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "cannot open file!\n");
        exit(1);
    }
    lseek(fd, 0, SEEK_SET);
    traverse_tree(fd);

    if (close(fd) == -1) {
        fprintf(stderr, "error while deleting descriptor!\n");
    }
    printf("\n");


    return 0;
}
