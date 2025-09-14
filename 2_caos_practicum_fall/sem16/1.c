#include <stdio.h>
#include <stdlib.h>


enum
{
    NTOREAD = 5
};

int
main(int argc, char *argv[]) 
{
    unsigned int cache_size = strtol(argv[2], NULL, 10); 
    unsigned int block_size = strtol(argv[3], NULL, 10); 
    unsigned int block_count = cache_size / block_size;
    unsigned int *cache = calloc(block_count, sizeof(*cache));
    unsigned int miss_count = 0;
    char mode1, mode2;
    unsigned int addr, size, value;
    while (scanf("%c %c %x %d %d\n", &mode1, &mode2, &addr, &size, &value) == 5) {
        unsigned int memory_block_num = addr / block_size + 1;
        unsigned int cache_block_num = memory_block_num % block_count;
        if (mode1 == 'R' && (!cache[cache_block_num] || cache[cache_block_num] != memory_block_num)) {
            if (cache[cache_block_num]) {
                ++miss_count;
            }
            cache[cache_block_num] = memory_block_num;
        }
    }
    free(cache);
    printf("%u\n", miss_count);

    return 0;
}
