#include <stdio.h>
#include <stdlib.h>


int64_t min(int64_t a, int64_t b) {
    return a < b ? a : b; 
}

int64_t 
calc_max_size(int64_t block_size, int64_t block_num_size, int64_t inode_direct_block_count)
{
    int64_t res1 = 0;
    int64_t direct_sum = 0;
    if (__builtin_smulll_overflow(inode_direct_block_count, block_size, &direct_sum)) {
        return -1;
    }
    int64_t first_ref_sum = 0, second_ref_sum = 0, third_ref_sum = 0;
    int64_t ref_count = block_size / block_num_size;
    if (__builtin_smulll_overflow(ref_count, block_size, &first_ref_sum)) {
        return -1;
    }
    int64_t ref_count_pow2;
    if (__builtin_smulll_overflow(ref_count, ref_count, &ref_count_pow2)) {
        return -1;
    }
    if (__builtin_smulll_overflow(ref_count_pow2, block_size, &second_ref_sum)) {
        return -1;
    }
    int64_t ref_count_pow3;
    if (__builtin_smulll_overflow(ref_count_pow2, ref_count, &ref_count_pow3)) {
        return -1;
    }
    if (__builtin_smulll_overflow(ref_count_pow3, block_size, &third_ref_sum)) {
        return -1;
    }
    int64_t res2 = 0;
    if (__builtin_saddll_overflow(direct_sum, first_ref_sum, &res)) {
        return res1;
    }
    if (__builtin_saddll_overflow(res, second_ref_sum, &res)) {
        return res1;
    }
    if (__builtin_saddll_overflow(res, third_ref_sum, &res)) {
        return res1;
    }
    return min(res1, res2);
}

int
main(void)
{
    int64_t block_size;
    int64_t block_num_size;
    int64_t inode_direct_block_count;
    scanf("%lld%lld%lld", &block_size, &block_num_size, &inode_direct_block_count);
    printf("%lld\n", calc_max_size(block_size, block_num_size, inode_direct_block_count));
    int64_t opt_block_num_size = 0, mx_size = 0; 
    for (int64_t i = 1; i <= 7; ++i) {
        int64_t res = calc_max_size(block_size, i, inode_direct_block_count);
        if (res == -1) {
            printf("%lld %lld\n", i, res);
            return 0;
        }
        if (mx_size < res) {
            mx_size = res;
            opt_block_num_size = i;
        } else {
            printf("%lld %lld\n", opt_block_num_size, mx_size);
            return 0;
        }
    }
    printf("%lld %lld\n", opt_block_num_size, mx_size);


    return 0;
}
