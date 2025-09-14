#include "3.cpp"
int main() {
    const std::vector<int> a{0, 1, 2};
    std::vector<int> b{4, 5, 2, 3, 6};
    process(a, b);
    for (auto i : b) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;


}   
