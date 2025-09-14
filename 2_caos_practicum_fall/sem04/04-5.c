#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>

static const mode_t CHMOD = 0600;

int
main(int argc, char *argv[])
{

    // open files
    int fin = open(argv[argc - 3], O_RDONLY);
    int fout = open(argv[argc - 2], O_WRONLY | O_TRUNC | O_CREAT, CHMOD);

    if (fin == -1 || fout == -1) {
        fprintf(stderr, "open file err\n");
        exit(1);
    }
    errno = 0;
    char *eptr = NULL;
    int MOD = strtol(argv[argc - 1], &eptr, 10);

    if (errno || *eptr || argv[argc - 1] == eptr || (int) MOD != MOD) {
        fprintf(stderr, "incorrect input\n");
        exit(1);
    }

    long long int sum = 0, counter = 1;
    unsigned char buf = 0;

    while (read(fin, &buf, sizeof(buf)) == sizeof(buf)) {

        for (int i = 0; i < CHAR_BIT; ++i) {
            sum += (counter % MOD) * (counter % MOD);
            sum %= MOD;

            if (buf & 1) {
                if (write(fout, &sum, __SIZEOF_INT__) != __SIZEOF_INT__) {
                    fprintf(stderr, "write err\n");
                    exit(1);
                }
            }
            buf >>= 1;
            ++counter;
        }
    }

    // close file
    if (close(fin) == -1 || close(fout) == -1) {
        fprintf(stderr, "close file err\n");
        exit(1);
    }
    return 0;
}
