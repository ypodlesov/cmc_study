#include <iostream>

class A {
  public:
    A() noexcept { std::cin >> a_; } 
    A(A &a) : A() { flag = true; a_ += a.a_; }
    ~A() noexcept { if (flag) std::cout << a_ << std::endl; }

  private:
    int a_{};
    bool flag{}; 
};

