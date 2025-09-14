#include <stdio.h>
#include <stdlib.h>

#define handle_error(msg) do { perror(msg); exit(EXIT_FAILURE); } while (0)

enum 
{
    A = 1103515245,
    C = 12345,
    M = 1 << 31
};

typedef struct RandomOperations RandomOperations;

typedef struct RandomGenerator 
{
    int cur_val;  
    RandomOperations *ops;
} RandomGenerator;

struct RandomOperations 
{
    void (*destroy) (RandomGenerator *);
    int (*next) (RandomGenerator *);
};

void 
destroy(RandomGenerator *randgen) 
{
    free(randgen->ops);
    free(randgen);
}

int
next(RandomGenerator *randgen)
{
    randgen->cur_val = ((long long) randgen->cur_val * A + C) % M;
    return randgen->cur_val;
}

RandomGenerator *
random_create(int seed) 
{
    RandomGenerator *res = malloc(sizeof(RandomGenerator));
    RandomOperations *ops = malloc(sizeof(RandomGenerator));
    if (!res || !ops) handle_error("malloc");

    ops->next = next;
    ops->destroy = destroy;
    res->cur_val = seed;
    res->ops = ops;
    return res;
}


