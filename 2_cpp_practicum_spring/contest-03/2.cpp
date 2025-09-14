#include <cmath>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>

namespace numbers {
    /*
class complex {
  public:
    complex(double re = 0.0, double im = 0.0) : re_{re}, im_{im} {}
    complex(const complex &z) = default;
    explicit complex(std::string);
    double re() const { return re_; }
    double im() const { return im_; }
    double abs2() const { return re_ * re_ + im_ * im_; }
    double abs() const { return sqrt(abs2()); }
    const complex & operator +=(const complex &z);
    const complex & operator -=(const complex &z);
    const complex & operator *=(const complex &z);
    const complex & operator /=(const complex &z);
    complex & operator ~() const { return *(new complex(re_, -im_)); }
    complex & operator -() const { return *(new complex(-re_, -im_)); }
    std::string to_string() const;

  private:
    double re_{}, im_{};
};

complex::complex(std::string s) {
    std::string re_str{}, im_str{};
    auto i = s.begin() + 1;
    for (; i < s.end() && *i != ','; ++i) {
        re_str.push_back(*i);
    }
    for (++i; i < s.end() && *i != ')'; ++i) {
        im_str.push_back(*i);
    }
    re_ = stod(re_str);
    im_ = stod(im_str); 
}

const complex & complex::operator +=(const complex &z) {
    re_ += z.re();
    im_ += z.im();
    return *this;
}
const complex & complex::operator -=(const complex &z) {
    re_ -= z.re();
    im_ -= z.im();
    return *this;
}

const complex & complex::operator *=(const complex &z) {
    auto re{re_}, im{im_};
    re_ = re * z.re() - im * z.im();
    im_ = re * z.im() + im * z.re();
    return *this;
}

const complex & complex::operator /=(const complex &z) {
    auto re{re_}, im{im_};
    re_ = (re * z.re() + im * z.im()) / z.abs2();
    im_ = (z.re() * im - re * z.im()) / z.abs2();
    return *this;
}

inline complex operator +(complex a, complex b) {
    a += b;
    return a;
}

inline complex operator -(complex a, complex b) {
    a -= b;
    return a;
}

inline complex operator *(complex a, complex b) {
    a *= b;
    return a;
}

inline complex operator /(complex a, complex b) {
    a /= b;
    return a;
}

std::string complex::to_string() const {
    std::ostringstream s;
    s << std::setprecision(10) << '(' << re_ << ',' << im_ << ')';
    return s.str();
} 
*/

class complex_stack {
  public:
    complex_stack() = default;
    complex_stack(const complex_stack &other); // deep copy constructor
    complex_stack(const complex_stack &other, const complex &num);
    complex_stack(const complex_stack &other, const size_t size);
    ~complex_stack() { delete[] stack_; }
    size_t size() const { return size_; } // get size
    complex operator [](const size_t idx) const { return *(stack_ + idx); }
    void swap(complex_stack &other);
    complex_stack & operator =(complex_stack other);
    complex_stack operator <<(const complex &num) const; // to add elem on top
    complex_stack operator ~() const; // to remove from top
    complex operator +() const { return stack_[size_-1]; }

  private:
    size_t size_{};
    complex *stack_{nullptr};
};


complex_stack::complex_stack(const complex_stack &other) {
    size_ = other.size();
    stack_ = new complex[size_];
    for (size_t i = 0; i < size_; ++i) stack_[i] = other[i];
}

complex_stack::complex_stack(const complex_stack &other, const complex &num) {
    size_ = other.size() + 1;
    stack_ = new complex[size_];
    for (size_t i = 0; i < size_ - 1; ++i) stack_[i] = other[i];
    stack_[size_ - 1] = num;
}

complex_stack::complex_stack(const complex_stack &other, const size_t size) {
    if (!size) {
        stack_ = nullptr;
        return;
    }
    size_ = size;
    stack_ = new complex[size_];
    for (size_t i = 0; i < size_; ++i) stack_[i] = other[i];
}

void complex_stack::swap(complex_stack &other) {
    std::swap(size_, other.size_);
    std::swap(stack_, other.stack_);
}

complex_stack & complex_stack::operator =(complex_stack other) {
    swap(other);
    return *this;
}

complex_stack complex_stack::operator <<(const complex &num) const {
    return complex_stack(*this, num);
}

complex_stack complex_stack::operator ~() const {
    return complex_stack(*this, size_ - 1);
}

}














