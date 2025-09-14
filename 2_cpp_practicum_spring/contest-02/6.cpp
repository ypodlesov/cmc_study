#include <iostream>
#include <string>

class StringView {
  public:
    StringView(std::string &, size_t, size_t) noexcept;
    StringView(const StringView &, size_t, size_t) noexcept;
    size_t length() const noexcept { return end_ - begin_; }
    size_t size() const noexcept { return end_ - begin_; }
    std::string str(size_t begin = 0, size_t count = std::string::npos) const;

  private:
    std::string &s_;
    size_t begin_{};
    size_t end_{};
};


StringView::StringView(std::string &s, size_t begin = 0, 
    size_t count = std::string::npos) noexcept : s_{s}, begin_{begin},
                                                 end_{begin + count} {}

StringView(const StringView &sv, size_t begin = 0, 
           size_t count = std::string::npos) noexcept :  
