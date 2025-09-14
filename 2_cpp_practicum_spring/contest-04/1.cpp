#include <vector>
#include <iostream>

void process (
        const std::vector<uint64_t> &a, 
        std::vector<uint64_t> &b, 
        unsigned int step) 
{
    auto ait = a.begin();
    auto bit = b.rbegin(); 
    for (; ait < a.end() && bit < b.rend(); ait += step, ++bit) {
        *bit += *ait;     
    }
}
