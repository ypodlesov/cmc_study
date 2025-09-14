#include <string>
#include <cmath>
#include <sstream>

class Rectangle : public Figure {
  public:
    Rectangle(double a = 0, double b = 0) noexcept : a_{a}, b_{b} {}
    static Rectangle * make(std::string &&s) {
        std::stringstream tmp;
        tmp << s << std::endl;
        double a{}, b{};
        tmp >> a >> b;
        return new Rectangle(a, b);
    }
    virtual double get_square() const override { return a_ * b_; }
    virtual ~Rectangle() {}

  private:
    double a_{}, b_{};
};

class Square : public Figure {
  public:
    Square(double a = 0) noexcept : a_{a} {}
    static Square * make(std::string &&s) {
        std::stringstream tmp;
        tmp << s << std::endl;
        double a{};
        tmp >> a;
        return new Square(a);
    }
    virtual double get_square() const override { return a_ * a_; }
    virtual ~Square() {}

  private:
      double a_{};
};

class Circle : public Figure {
  public:
    Circle(double r = 0) noexcept : r_{r} {}
    static Circle * make(std::string &&s) {
        std::stringstream tmp;
        tmp << s << std::endl;
        double r{};
        tmp >> r;
        return new Circle(r);
    }
    virtual double get_square() const override { return M_PI * r_ * r_; }
    virtual ~Circle() {}

  private:
      double r_{};
};
