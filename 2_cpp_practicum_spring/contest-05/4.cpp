#include <iostream>
#include <map>
#include <vector>

constexpr unsigned int kMod = 4294967161;

int main() {

    std::map<std::pair<unsigned int,unsigned int>, long long> matrix{};
    for (int t{2}; t--;) {
        unsigned int R{}, C{}, V{};
        while (std::cin >> R >> C >> V && !(R == 0 && C == 0 && V == kMod)) {
            matrix[std::make_pair(R, C)] += V;
            matrix[std::make_pair(R, C)] %= kMod;
        }
    }
    for (auto i : matrix) {
        if (!i.second) continue;
        std::cout << i.first.first << ' ' << i.first.second << ' ' << i.second;
        std::cout << std::endl;
    }

}
