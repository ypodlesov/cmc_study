#include <vector>
#include <functional>
#include <iterator>

template <typename It, typename F>
void myapply(It begin, It end, F func) {
    for (auto it = begin; it != end; ++it) func(*it);
}

template <typename It, typename P>
std::vector<std::reference_wrapper
           <typename std::iterator_traits<It>::value_type>> 
myfilter2(It begin, It end, P pred) {
    std::vector<std::reference_wrapper
        <typename std::iterator_traits<It>::value_type>> res{};
    for (auto it = begin; it != end; ++it) {
        if (pred(*it)) res.push_back(*it);
    }
    return res;
}


//int main() {
//    int a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
//    auto r = myfilter2(a1+2, a1+10, [](auto z) { return !(int(z) & 1); });    
//}

