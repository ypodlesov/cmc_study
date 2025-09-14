#include <cmath>
#include <iomanip>
#include <sstream>

namespace numbers
{
    class complex
    {
        double _im, _re;
    public:
        complex (double re = 0, double im = 0) {
            _im = im;
            _re = re;
        }
        complex (const complex &a) {
            _im = a.im();
            _re = a.re();
        }
        complex (std::string s) {
            std::string a = "";
            for (size_t i = 0; i < s.size(); i++) {
                if (s[i] == ',' || s[i] == ')' || s[i] == '(') {
                    if (s[i] != '(') {
                        if (s[i] == ',') {
                            _re = std::stod(a);
                        } else {
                            _im = std::stod(a);
                        }
                    }
                    a = "";
                } else {
                    a += s[i];
                }
            }
        }
        double re() const
        {
            return _re;
        }
        double im() const
        {
            return _im;
        }
        double abs2() const
        {
            return _re * _re + _im * _im;
        }
        double abs() const
        {
            return sqrt(abs2());
        }
        complex& operator+=(const complex& a)
        {
            _re += a.re();
            _im += a.im();
            return *this;
        }
        complex& operator-=(const complex& a)
        {
            _re -= a.re();
            _im -= a.im();
            return *this;
        }
        complex& operator*=(const complex& a)
        {
            double re = a.re(), im = a.im();
            double re1 = _re, im1 = _im;
            _re = re1 * re - im1 * im;
            _im = re1 * im + im1 * re;
            return *this;
        }
        complex& operator/=(const complex& a)
        {
            double re = a.re(), im = a.im();
            double re1 = _re, im1 = _im;
            _re = (re1 * re + im1 * im) / a.abs2();
            _im = (im1 * re - re1 * im) / a.abs2();
            return *this;
        }
        std::string to_string() const
        {
            std::ostringstream str;
            str << std::setprecision(10) << '(' << _re << ',' << _im << ')';
            return str.str();
        }
        complex& operator~() const
        {
            complex *a = new complex(_re, -_im);
            return *a;
        }
        complex& operator-() const
        {
            complex *a = new complex(-_re, -_im);
            return *a;
        }
        complex& operator=(const complex &a)
        {
            if (this != &a) {
                _im = a.im();
                _re = a.re();
            }
            return *this;
        }
    };
    inline complex operator+(complex a, complex b)
    {
        a += b;
        return a;
    }
    inline complex operator-(complex a, complex b)
    {
        a -= b;
        return a;
    }
    inline complex operator*(complex a, complex b)
    {
        a *= b;
        return a;
    }
    inline complex operator/(complex a, complex b)
    {
        a /= b;
        return a;
    }
}
