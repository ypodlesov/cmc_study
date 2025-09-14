// здесь могут быть нужные Вам директивы #include

#include <string>
#include <algorithm>

class BinaryNumber
{
  public:
    BinaryNumber(const std::string &s) noexcept {
        s_.resize(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            s_[i] = s[i];
        }
    }
    operator std::string () const { return s_; };
    const BinaryNumber &operator++() {
    std::reverse(s_.begin(), s_.end());
    size_t n = 0;
    while (n < s_.size() && s_[n] != '0') {
        s_[n] = '0';
        ++n;
    }
    if (n < s_.size()) {
        s_[n] = '1';
    } else {
        s_.insert(s_.end(), '1');
    }
    std::reverse(s_.begin(), s_.end());
    return *this;
}
  private:
    std::string s_{}; 
};


