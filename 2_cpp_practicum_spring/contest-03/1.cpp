#include <cmath>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace numbers {
class complex {
  public:
    complex(double re = 0.0, double im = 0.0) : re_{re}, im_{im} {}
    complex(const complex &z) : re_{z.re()}, im_{z.im()} {};
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
}


