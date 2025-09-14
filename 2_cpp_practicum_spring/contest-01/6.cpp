#include <iostream>
#include <cmath>
#include <ios>
#include <iomanip>

constexpr double kEps = 1e-9;

namespace Geom2D {
class Point {
  public:
    [[nodiscard]] double x() const noexcept { return x_; }
    [[nodiscard]] double y() const noexcept { return y_; }
    void set_x(double val) noexcept { x_ = val; }
    void set_y(double val) noexcept { y_ = val; }
    Point() noexcept = default;
    Point (double x, double y) noexcept : x_{x}, y_{y} {}

  private:
    double x_{}, y_{};
};

class Line {
  public:
    Line (const Point &p1, const Point &p2) noexcept { 
        a_ = p1.y() - p2.y();
        b_ = p2.x() - p1.x();
        c_ = p1.x() * p2.y() - p2.x() * p1.y();
    }    
    [[nodiscard]] double a() const noexcept { return a_; }
    [[nodiscard]] double b() const noexcept { return b_; }
    [[nodiscard]] double c() const noexcept { return c_; }

  private:
    double a_{}, b_{}, c_{};
};

class Vector {
  public: 
    Vector() = default;
    Vector(double x, double y) noexcept : x_{x}, y_{y} {}
    Vector(const Point &a, const Point &b) noexcept : 
                                        Vector(b.x() - a.x(), b.y() - a.y()) {} 
    [[nodiscard]] double x() const noexcept { return x_; }
    [[nodiscard]] double y() const noexcept { return y_; }
    [[nodiscard]] double abs2() const { return x_ * x_ + y_ * y_; }
    [[nodiscard]] double abs() const { return sqrt(this->abs2()); }

  private:
    double x_{}, y_{};
};

std::istream & operator>>(std::istream &is, Point &p) {
    double x, y;
    is >> x >> y;
    p.set_x(x), p.set_y(y);
    return is;
}

std::ostream & operator<<(std::ostream &os, const Point &p) {
    os << std::fixed << std::setprecision(5);
    os << p.x() << ' ' << p.y();
    return os;
}

bool AreEqual(const double a, const double b) {
    return std::abs(a - b) < kEps;
}

bool AreParallel(const Line &l1, const Line &l2) {
    return (AreEqual(l1.a() * l2.b(), l1.b() * l2.a()));
}

double DotProduct(const Vector &v1, const Vector &v2) {
    return v1.x() * v2.x() + v1.y() * v2.y();
}

bool AreSame(const Line &l1, const Line &l2) {
    Vector v1(l1.b(), -l1.a());
    Vector v2(l2.b(), -l2.a());
    return AreEqual(l1.a() * l2.b(), l2.a() * l1.b()) && 
           AreEqual(l1.b() * l2.c(), l2.b() * l1.c()) && 
           AreEqual(l1.a() * l2.c(), l2.a() * l1.c());
}

Point IntersectionPoint(const Line &l1, const Line &l2) {
    double det = l1.a() * l2.b() - l2.a() * l1.b();
    return Point((l1.b() * l2.c() - l2.b() * l1.c()) / det, 
            (l2.a() * l1.c() - l1.a() * l2.c()) / det);
}
};




int main() {

    Geom2D::Point a, b, c, d;
    std::cin >> a >> b >> c >> d;
    Geom2D::Line l1(a, b);
    Geom2D::Line l2(c, d);
    if (Geom2D::AreSame(l1, l2)) {
        std::cout << 2 << std::endl;
    } else if (!Geom2D::AreParallel(l1, l2)) {
        std::cout << 1 << ' ' << Geom2D::IntersectionPoint(l1, l2) << std::endl;
    } else {
        std::cout << 0 << std::endl;
    }

    return 0;
}

