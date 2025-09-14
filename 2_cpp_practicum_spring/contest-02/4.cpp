#include <iostream>
#include <iomanip>

int main() {
    char c{}, prev{'\0'};
    int times{};
    while (std::cin.get(c)) {
        if (c != prev) {
            if (times > 4 || prev == '#') {
                std::cout << '#' << prev << std::hex << times << '#';
            } else {
                for (int i = 0; i < times; ++i) {
                    std::cout << prev;
                }
            }
            prev = c;
            times = 1;
        } else {
            ++times;
        }
    }
    if (times > 4 || prev == '#') {
        std::cout << '#' << prev << std::hex << times << '#';
    } else {
        for (int i = 0; i < times; ++i) {
            std::cout << prev;
        }
    }

}