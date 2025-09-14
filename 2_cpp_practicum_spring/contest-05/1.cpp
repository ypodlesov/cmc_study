#include <iostream>
#include <vector>
#include <algorithm>
//#include <pair>

int main() {
        
    std::vector<std::pair<unsigned int, int>> nums{};
    for (int cur{}, idx{}; std::cin >> cur; ++idx) nums.emplace_back(cur, idx);
    std::stable_sort(nums.begin(), nums.end(), [](auto a, auto b) {
        int cnt_a{}, cnt_b{};
        for (int i = 0; i <= 31; ++i) {
            cnt_a += a.first & (1 << i) ? 1 : 0;
            cnt_b += b.first & (1 << i) ? 1 : 0;
        }
        return cnt_a < cnt_b || (cnt_a == cnt_b && a.second < b.second);
    });
    for (auto i : nums) std::cout << i.first << std::endl;

    return 0;
}
