#include <algorithm>
#include <iostream>
#include <vector>

void process(const std::vector<int> &a, std::vector<int> &b) {
    std::vector<int> tmp = a;
    std::sort(tmp.rbegin(), tmp.rend());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    for (auto it = tmp.begin(); it < tmp.end(); ++it) {
        if (*it >= int(b.size()) || *it < 0) continue;
        b.erase(b.begin() + *it);    
    }
    b.shrink_to_fit();
}
