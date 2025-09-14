#pragma once
#include <iostream>

template <typename T>
class TMatrix;

template <typename T>
class TVector {
public:
    TVector() = default;
    TVector(size_t size);
    TVector(const TVector& other);
    TVector(TVector&& other) noexcept;
    TVector(const TMatrix<T>& a, size_t j);
    TVector operator =(const TVector& other);
    TVector operator =(TVector&& other) noexcept;

    T* GetData() const noexcept;
    size_t GetSize() const noexcept;

    bool operator !() const noexcept;
    T& operator ()(size_t i) const;
    TVector& operator +=(const TVector& other);
    TVector& operator -=(const TVector& other);
    template <typename DType>
    TVector& operator *=(const DType coeff);

    bool Nullify() noexcept;

    static double Norm2(const TVector& v);
    static double InnerProd(const TVector& v, const TVector& u);
    static TVector CreateRandom(const size_t size);

    ~TVector();

private:
    size_t Size{};
    T* Data;
};

template <typename T>
std::ostream& operator <<(std::ostream& out, const TVector<T>& v);

template <typename T>
bool operator ==(const TVector<T>& a, const TVector<T>& b);

template <typename T>
bool operator !=(const TVector<T>& a, const TVector<T>& b);

template <typename T>
TVector<T> operator +(const TVector<T>& a, const TVector<T>& b);

template <typename T>
TVector<T> operator -(const TVector<T>& a, const TVector<T>& b);

template <typename T1, typename T2>
TVector<T1> operator *(const TVector<T1>& a, const T2 coeff);

template <typename T>
T InnerProd(const TVector<T>& a, const TVector<T>& b);
