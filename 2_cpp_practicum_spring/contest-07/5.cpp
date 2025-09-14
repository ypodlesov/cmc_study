#include <string>
#include <cmath>
#include <sstream>

class Figure {
  public:
    virtual double get_square() const = 0;
    virtual bool equals(Figure *) const = 0;
    virtual ~Figure() {}

  private:

};

class Rectangle : public Figure {
  public:
    


  private:
    double a_{}, b_{};
};

class Triangle : public Figure {
  public:
    
  private:
}
