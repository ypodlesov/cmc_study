// #include <vector>
// #include <iostream>
// 
// struct F {
//     bool operator ()(int a) {
//         return a <= 3;
//     }
// };
// 
// bool f(int a) {
//     return a <= 3;
// }

template <typename T, typename P>
T myfilter(const T &a, P pred) {
    T res{};
    for (const auto &i : a) {
        if (pred(i)) {
            res.insert(res.end(), i);
        }
    }
    return res;
}

// int main() {
//     std::vector<int> a{1, 2, 3, 4, 5};
//     //F f{};
//     std::vector<int> b = myfilter(a, f);
//     for (auto i : b) std::cout << i << ' ';
//     std::cout << std::endl;
// 
// }

