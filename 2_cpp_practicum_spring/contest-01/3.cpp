#include <iostream>

int main() {

    std::ios_base::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);
    char c{'\n'};
    int flag{};
    while (std::cin.get(c)) {
        if (isdigit(c)) {
            if (flag == 2) std::cout << c;
            if (c == '0' && flag == 0) {
                flag = 1;
                continue;
            }
            if (c == '0' && flag == 1) continue;
            if (flag < 2) {
                flag = 2;
                std::cout << c;
            }
        } else {
            if (flag == 1) std::cout << '0';
            std::cout << c;
            flag = 0;
        }
    }
    if (flag == 1) std::cout << '0';

    return 0;
}