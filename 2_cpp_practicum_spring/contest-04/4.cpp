#include <iostream>
#include <vector>
#include <algorithm>

class functor {

  public:
    int sum() noexcept { return sum_; }
    int operator() (const int value) const{
        return sum_ += value;   //возводим в квадрат
    }
  private:
    int sum_
};

int main() {
    std::vector<int> a;
    
}
