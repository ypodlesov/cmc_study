#include "1.cpp"

int main() {

    std::vector<unsigned long long> a{1, 2, 3, 4};
    std::vector<unsigned long long> b{1, 2, 3};
    process(a, b, 2);
    for (auto i : b) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;


    return 0;
}
