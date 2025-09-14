#include <iostream>
#include <vector>

void process(std::vector<int64_t> &a, int64_t margin) {
    a.reserve(a.size() * 2);
    for (auto it = a.rbegin(); it < a.rend(); ++it) {
        if (*it >= margin) {
            a.push_back(*it);
        }
    }
    a.shrink_to_fit();
}
