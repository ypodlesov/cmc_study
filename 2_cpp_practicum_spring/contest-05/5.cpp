#include <iostream>
#include <map>

constexpr unsigned int kMod = 4294967161;

int main() {

    std::map<unsigned int, std::map<unsigned int, unsigned long long>> matrix1{};
    std::map<unsigned int, std::map<unsigned int, unsigned long long>> matrix2{};
    std::map<unsigned int, std::map<unsigned int, unsigned long long>> res{};

    for (unsigned int R{}, C{}, V{}; std::cin >> R >> C >> V && 
            !(R == 0 && C == 0 && V == kMod);) {
        matrix1[R][C] = V;
        matrix1[R][C] %= kMod;
    }
    for (unsigned int R{}, C{}, V{}; std::cin >> R >> C >> V && 
            !(R == 0 && C == 0 && V == kMod);) {
        matrix2[R][C] = V;
        matrix2[R][C] %= kMod;
    }
    for (auto i : matrix1) {
        for (auto j : i.second) {
            for (auto k : matrix2[j.first]) {
                unsigned long long tmp = j.second * matrix2[j.first][k.first];
                tmp %= kMod;
                res[i.first][k.first] = (res[i.first][k.first] + tmp) % kMod; 
            }
        }
    }
    for (auto i : res) {
        for (auto j : i.second) {
            if (!j.second) continue;
            std::cout << i.first << ' ' << j.first << ' ' << j.second;
            std::cout << std::endl;
        }
    }

    
}
