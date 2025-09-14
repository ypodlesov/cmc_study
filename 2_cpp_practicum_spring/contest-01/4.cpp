#include <iostream>
#include <iomanip>
#include <cmath>

int main() {

    std::cin.tie(0);
    long double a{};
    long double sum{};
    long double res{};
    int n{};
    while (std::cin >> a) {
        ++n;
        long double tmp{sum};
        sum += (a - sum) / n;
        res += (a - sum) * (a - tmp);
    }
    std::cout << std::fixed << std::setprecision(10);
    if (!n)  std::cout << 0 << 0 << std::endl;
    else std::cout << sum << ' ' << sqrt(res / n) << std::endl;

    return 0;
}