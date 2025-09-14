// #include <iostream>
// #include <vector>

template <typename T>
typename T::value_type process(const T &a) {
    typename T::value_type res{};
    if (a.end() == a.begin()) return res;
    auto it = a.end();
    res += *(--it); // it = a.end() - 1
    if (it-- == a.begin()) return res; // it = a.end() - 2
    if (it-- == a.begin()) return res; // it = a.end() - 3
    res += *(it);
    if (it-- == a.begin()) return res; // it = a.end() - 4
    if (it-- == a.begin()) return res; // it = a.end() - 5
    res += *(it);
    return res;
}

// int main() {
//     std::vector<int> a{1, 2, 3, 4, 5};
//     std::cout << process(a) << std::endl;
// }

