class Sum {
  public:
    Sum(int a, int b) noexcept : a_{a}, b_{b} {}
    Sum(Sum a, int b) noexcept : Sum{a.get(), b} {}
    [[nodiscard]] int get() const noexcept { return a_ + b_; }

  private:
    int a_{}, b_{};
};





