#include <iostream>
#include <map>
#include <vector>

int main() {

    std::map<std::string, std::vector<int>> students{};
    std::string surname{}; 
    int score{};
    for (; std::cin >> surname >> score; ) {
        students[surname].push_back(score);
    }
    for (auto i : students) {
        double sum{};
        for (auto j : i.second) sum += j;
        std::cout << i.first << ' ' << sum / i.second.size() << std::endl;
    }

}
