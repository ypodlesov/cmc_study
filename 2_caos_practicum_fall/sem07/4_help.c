#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int fd = open(argv[1], O_CREAT | O_TRUNC | O_RDWR);
    printf("%d\n", fd);
    int v;
    while ((scanf("%d", &v)) != EOF) {
        write(fd, &v, sizeof(v));
    }
    close(fd);

    return 0;
}
