#include <ranges>
#include <vector>
#include <iostream>

class Matrix {
  public:
    Matrix() noexcept;
    int & operator [](int row, int col); 
    auto begin() noexcept { return m_.begin(); }
    auto end() noexcept { return m_.end(); }
    
  private:
    int row_size{3};
    int col_size{3};
    std::vector<std::vector<int>> m_{};
};

int & Matrix::operator [](int row, int col) {
    if (row >= col_size || col >= row_size) { 
        throw std::out_of_range("out of bounds\n"); 
    }
    return m_[row][col];
}   

inline Matrix::Matrix() noexcept {
    m_.resize(col_size);
    for (int i = 0; i < col_size; ++i) {
        m_[i].resize(row_size);
    }
}

