#include <iostream>
#include <sstream>
#include <cctype>

enum { OFFSETSZ = 6, NUMBITS = 32, SHIFT = 4, NUMPERSTR = 4 };

unsigned int to_uint(char c) {
    if (isdigit(c)) {
        return c - '0';
    } else {
        return 10 + c - 'a';
    }
} 

int main() {

    std::ios_base::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);
    char c{};
    unsigned int offset{}, num{}, cnt_bits{}, cnt_nums{};
    while (std::cin >> c) {
        if (cnt_nums == NUMPERSTR) {
            cnt_nums = 0;
            offset = 0;
        }
        if (offset < OFFSETSZ) {
            ++offset;
            continue;          
        }
        if (isspace(c)) {
            continue;
        }
        cnt_bits += SHIFT;
        num <<= SHIFT;
        num += to_uint(c);
        if (cnt_bits == NUMBITS) {
            std::cout << num << std::endl;
            num = 0;
            cnt_bits = 0;
            ++cnt_nums;
        }
    }
    if (cnt_bits == NUMBITS) {
        std::cout << num << std::endl;
    }

    return 0;
}

