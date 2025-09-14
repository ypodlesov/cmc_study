#include "2.cpp"

int main() {
    std::vector<int64_t> a{1, 4, 3, 2};
    int64_t m = 3;
    process(a, m);
    for (auto i : a) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
}
