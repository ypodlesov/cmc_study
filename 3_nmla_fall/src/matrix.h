#pragma once
#include <iostream>

enum class ETriangularType {
    Upper,
    Lower
};

template <typename T>
class TVector;

#pragma once

template <typename T>
class TMatrix {
public:
    TMatrix() = default;
    TMatrix(size_t size1, size_t size2);
    TMatrix(size_t size);
    TMatrix(const TMatrix& other);
    TMatrix(TMatrix&& other) noexcept;
    TMatrix& operator =(const TMatrix& other);
    TMatrix& operator =(TMatrix&& other) noexcept;

    T* GetData() const noexcept;
    size_t GetSize1() const noexcept;
    size_t GetSize2() const noexcept;

    bool operator !() const noexcept;
    T& operator ()(size_t row, size_t column) const;
    TMatrix& operator +=(const TMatrix& other);
    TMatrix& operator -=(const TMatrix& other);
    template <typename DType>
    TMatrix& operator *=(const DType coeff);
    operator double() const;

    void Nullify();
    TMatrix Transpose() const;
    TMatrix CreateFromRange(size_t row1, size_t row2, size_t col1, size_t col2) const; // [row1,row2), [col1, col2)
    void AssignBlock(TMatrix& matrixBlock, size_t row1, size_t row2, size_t col1, size_t col2); // [row1,row2), [col1, col2)
    void AssignColumn(size_t columnIdx, const TVector<T>& v);
    std::pair<T, T> GetSpectrumBoundaries() const;

    static bool IsTriangular(const TMatrix& a, ETriangularType type);
    static double Norm2(const TMatrix& a);
    static double ColumnNorm2(const TMatrix& a, size_t colNum);
    static bool AbleToMultiply(const TMatrix& a, const TMatrix& b);
    static TMatrix<TMatrix> MakeBlockMatrix(const TMatrix& a, const std::pair<size_t, size_t>& blockSize);
    static TMatrix Prod(const TMatrix& a, const TMatrix& b);
    static TMatrix ParallelProd(const TMatrix& a, const TMatrix& b);
    static TMatrix BlockProd(const TMatrix& a, const TMatrix& b, const std::tuple<size_t, size_t, size_t>& blockSizes);
    static TMatrix CreateIdentityMatrix(const size_t n);
    static TMatrix CreateRandom(const size_t size1, const size_t size2);
    static T InnerProd(const TMatrix& a, size_t a_column, const TMatrix& b, size_t b_column);
    static TMatrix FromBlockMatrix(const TMatrix<TMatrix>& a);


    ~TMatrix();
    
private:
    size_t Size1{};
    size_t Size2{};
    T* Data{nullptr};
};

template <typename T>
std::ostream& operator <<(std::ostream& out, const TMatrix<T>& matrix);

template <typename T>
bool operator ==(const TMatrix<T>& a, const TMatrix<T>& b);

template <typename T>
bool operator !=(const TMatrix<T>& a, const TMatrix<T>& b);

template <typename T>
TMatrix<T> operator +(const TMatrix<T>& a, const TMatrix<T>& b);

template <typename T>
TMatrix<T> operator -(const TMatrix<T>& a, const TMatrix<T>& b);

template <typename T1, typename T2>
TMatrix<T1> operator *(const TMatrix<T1>& a, const TMatrix<T2>& b);

template <typename T1, typename T2>
TMatrix<T1> operator *(const TMatrix<T1>& a, const T2 coeff);

template <typename T>
TVector<T> operator *(const TMatrix<T>& a, const TVector<T>& v);