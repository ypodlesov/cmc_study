#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


#define handle_error(msg) do { perror(msg); exit(EXIT_FAILURE); } while (0)
#define MAXCOL(j, N) ((j) == (N))
#define MINCOL(j, N) ((j) == (N))
#define MAXROW(i, M) ((i) == (M))
#define MINROW(i, M) ((i) == (M))

enum { RIGHT, LEFT, UP, DOWN, MODE = 0600 };

int
main(int argc, char *argv[]) 
{
    int m = strtol(argv[2], NULL, 10);
    int n = strtol(argv[3], NULL, 10);
    int fd = open(argv[1], O_RDWR | O_CREAT | O_TRUNC, MODE);
    if (fd < 0) handle_error("open");
    if (ftruncate(fd, n * m * sizeof(int))) handle_error("ftruncate");
    int *a = mmap(NULL, n * m * sizeof(*a), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (a == MAP_FAILED) handle_error("mmap");

    // fill the table
    int maxcol = n - 1;
    int mincol = 0;
    int maxrow = m - 1;
    int minrow = 0;
    int mode = 0;
    int i = 0, j = 0;
    for (int k = 1; k <= n * m; ++k) {
        if (mode == RIGHT && MAXCOL(j, maxcol)) {
            mode = DOWN;  
            ++minrow;
        } else if (mode == LEFT && MINCOL(j, mincol)) {
            mode = UP;
            --maxrow;
        } else if (mode == UP && MINROW(i, minrow)) {
            mode = RIGHT;
            ++mincol;
        } else if (mode == DOWN && MAXROW(i, maxrow)) {
            mode = LEFT;
            --maxcol;
        }
        a[i * n + j] = k;
        switch(mode) {
            case RIGHT:
                ++j;
                break;
            case LEFT:
                --j;
                break;
            case UP:
                --i;
                break;
            case DOWN:
                ++i;
                break;
        }
    }
    //for (int i = 0; i < n * m; ++i) {
    //    if (i && i % n == 0) printf("\n");
    //    printf("%ld ", a[i]);
    //}
    munmap(a, n * m * sizeof(*a));
    close(fd);

    return 0;
}










