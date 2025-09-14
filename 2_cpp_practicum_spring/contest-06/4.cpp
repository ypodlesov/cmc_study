#include <iostream>
#include <algorithm>

template <typename It1, typename It2>
It2 myremove(It1 begin1, It1 end1, It2 begin2, It2 end2) {
    std::sort(begin1, end1, std::greater<>());
    end1 = std::unique(begin1, end1);
    auto it1 = end1;
    --it1;
    size_t sz2{};
    for (auto it = begin2; it != end2; ++it, ++sz2) {}
    while (sz2 > 0) {
        if ((*it1) >= sz2) {
            if (it1 == begin1) break;
            --it1;   
            continue; 
        }
        
        --sz2;
        --end2;
        size_t idx{};
        auto it{begin2};
        while (idx < *it1) {
            ++idx; ++it;
        }
        while (it != end2) {
            auto prev = it++;
            std::swap(*prev, *it);
        }

        if (it1 == begin1) break;
        --it1;
    }
    return end2;


}

