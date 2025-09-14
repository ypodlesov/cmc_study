#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VINDENT printf("=================\n");

const int MODE = 0666;
const int n = 10;

typedef struct Node {
    int32_t key;
    int32_t left_idx;
    int32_t right_idx;
} Node;

int 
min(int a, int b) 
{
    return a ? a < b : b;
}

int 
max(int a, int b) 
{
    return a ? a > b : b;
}


int 
main(int argc, char *argv[]) 
{

    srand(time(NULL));
    if (argc < 2) {
        printf("error!\n");
        exit(1);
    }
    char *path = argv[1];
    int fd = open(path, O_RDWR | O_TRUNC | O_CREAT, MODE);
    struct Node *array = calloc(n, sizeof(Node));
    for (int i = 0; i < n; ++i) {
        array[i].key = i;
        if (i * 2 + 1 < n) {
            array[i].left_idx = i * 2 + 1;
        } else {
            array[i].left_idx = 0;
        }
        if (i * 2 + 2 < n) {
            array[i].right_idx = i * 2 + 2;
        } else {
            array[i].right_idx = 0;
        }
        printf("%d %d %d\n", array[i].key, array[i].left_idx, array[i].right_idx);
        write(fd, &array[i], sizeof(Node));
    } 
    close(fd);

    return 0;
}

// ssize_t read(int fildes, void *buf, size_t nbyte);
// ssize_t write(int fildes, const void *buf, size_t nbyte);
// off_t lseek(int fildes, off_t offset, int whence);

