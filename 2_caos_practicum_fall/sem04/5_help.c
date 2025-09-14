#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


int main() {

    int fd = open("in", O_RDWR | O_TRUNC | O_CREAT, 0666);
    unsigned int n = 0x11111111;
    write(fd, &n, sizeof(n));

    return 0;
}
