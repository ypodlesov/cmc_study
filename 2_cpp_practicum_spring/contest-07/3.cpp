#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <algorithm>

int main() {
    std::string s{};
    std::vector<std::pair<double, std::string>> a{};
    while (std::getline(std::cin, s)) {
        auto it = s.begin();
        while (it != s.end() && !isalpha(*it)) ++it;
        if (it == s.end()) continue;
        auto type = *it;
        s.erase(s.begin(), it + 1);
        std::unique_ptr<Figure> ptr{};
        if (type == 'R') {
            ptr = static_cast<std::unique_ptr<Figure>>(Rectangle::make(s));
        }
        if (type == 'S') {
            ptr = static_cast<std::unique_ptr<Figure>>(Square::make(s));
        }
        if (type == 'C') {
            ptr = static_cast<std::unique_ptr<Figure>>(Circle::make(s));
        }
        a.emplace_back(ptr->get_square(), ptr->to_string());
    }
    std::stable_sort(a.begin(), a.end(), 
            [](const auto &a, const auto &b) { return a.first < b.first; });
    for (const auto &i : a) {
        std::cout << i.second << std::endl;
    }
}
