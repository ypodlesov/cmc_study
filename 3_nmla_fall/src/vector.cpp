#include "common.h"
#include "matrix.h"
#include "vector.h"

#include <cmath>
#include <cstdlib>
#include <ctime>

template <typename T>
TVector<T>::TVector(size_t size) 
    : Size{size}
    {
        Data = new T[Size]; 
    }

template <typename T>
TVector<T>::TVector(const TVector& other) 
    : TVector<T>(other.Size)
    {
    }

template <typename T>
TVector<T>::TVector(TVector&& other) noexcept 
    : Size{other.Size} 
    {
        Data = other.Data;
        other.Data = nullptr;
    }

template <typename T>
TVector<T>::TVector(const TMatrix<T>& a, size_t j) 
    : Size{a.GetSize1()}
    {
        Data = new T[Size];
        for (size_t i = 0; i < Size; ++i) {
            Data[i] = a(i, j);
        }
    }

template <typename T>
TVector<T> TVector<T>::operator =(const TVector& other) {
    return TVector(other.Size);
}

template <typename T>
TVector<T> TVector<T>::operator =(TVector&& other) noexcept {
    Size = other.Size;
    Data = other.Data;
    other.Data = nullptr;
    return *this;
}

template <typename T>
T* TVector<T>::GetData() const noexcept {
    return Data;
}

template <typename T>
size_t TVector<T>::GetSize() const noexcept {
    return Size;
}

template <typename T>
bool TVector<T>::operator !() const noexcept {
    return !Data;
}

template <typename T>
T& TVector<T>::operator ()(size_t i) const {
    if (!Data) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    if (i > Size) {
        throw(std::out_of_range("Cannot get vector element."));
    }
    return Data[i];
}

template <typename T>
TVector<T>& TVector<T>::operator +=(const TVector& other) {
    if (!Data || !other.Data) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    if (Size != other.GetSize()) {
        throw(std::invalid_argument("Vectors sizes are different. Cannot apply +=."));
    }
    for (size_t i = 0; i < Size; ++i) {
        Data[i] += other(i);
    }
    return *this;
}

template <typename T>
TVector<T>& TVector<T>::operator -=(const TVector& other) {
    if (!Data || !other.Data) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    if (Size != other.GetSize()) {
        throw(std::invalid_argument("Vectors sizes are different. Cannot apply -=."));
    }
    for (size_t i = 0; i < Size; ++i) {
        Data[i] -= other(i);
    }
    return *this;
}

template <typename T> template <typename DType>
TVector<T>& TVector<T>::operator *=(const DType coeff) {
    if (!Data) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    for (size_t i = 0; i < Size; ++i) {
        Data[i] *= coeff;
    }
    return *this;
}

template <typename T>
bool TVector<T>::Nullify() noexcept {
    if (!Data) {
        return false;
    }
    for (size_t i = 0; i < Size; ++i) {
        Data[i] = 0;
    }
    return true;
}

template <typename T>
double TVector<T>::Norm2(const TVector<T>& v) {
    if (!v) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    double norm = 0;
    size_t size = v.GetSize();
    for (size_t i = 0; i < size; ++i) {
        norm += v(i) * v(i);
    }
    return std::sqrt(norm);
}

template <typename T>
double TVector<T>::InnerProd(const TVector<T>& v, const TVector<T>& u) {
    double res = 0.0;
    for (size_t i = 0; i < v.GetSize(); ++i) {
        res += v(i) * u(i);
    }
    return res;
}

template <typename T>
TVector<T> TVector<T>::CreateRandom(const size_t size) {
    std::srand(std::time(nullptr));
    TVector<T> v(size);
    for (size_t i = 0; i < size; ++i) {
        int tmp = std::rand();
        v(i) = static_cast<double>(tmp) / RAND_MAX * std::pow(-1, (tmp & 1));
    }
    return v;
}

template <typename T>
TVector<T>::~TVector() {
    delete [] Data;
    Size = 0;
}


template <typename T>
std::ostream& operator <<(std::ostream& out, const TVector<T>& v) {
    if (!v) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    size_t size = v.GetSize();
    for (size_t i = 0; i < size; ++i) {
        out << v(i) << ' ';
    }
    return out;
}

template <typename T>
bool operator ==(const TVector<T>& a, const TVector<T>& b) {
    if (!a || !b) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    if (a.GetSize() != b.GetSize()) {
        throw(std::invalid_argument("Vectors sizes are different. Cannot apply +."));
    }
    size_t size = a.GetSize();
    for (size_t i = 0; i < size; ++i) {
        if (!RoughEq(a(i), b(i))) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool operator !=(const TVector<T>& a, const TVector<T>& b) {
    if (!a || !b) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    if (a.GetSize() != b.GetSize()) {
        throw(std::invalid_argument("Vectors sizes are different. Cannot apply +."));
    }
    return !(a == b);
}

template <typename T>
TVector<T> operator +(const TVector<T>& a, const TVector<T>& b) {
    if (!a || !b) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    if (a.GetSize() != b.GetSize()) {
        throw(std::invalid_argument("Vectors sizes are different. Cannot apply +."));
    }
    TVector res(a);
    return res += b;
}

template <typename T>
TVector<T> operator -(const TVector<T>& a, const TVector<T>& b) {
    if (!a || !b) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    if (a.GetSize() != b.GetSize()) {
        throw(std::invalid_argument("Vectors sizes are different. Cannot apply +."));
    }
    TVector res(a);
    return res += b;
}

template <typename T1, typename T2>
TVector<T1> operator *(const TVector<T1>& a, const T2 coeff) {
    if (!a) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    TVector res(a);
    return res *= coeff;
}

template <typename T>
T InnerProd(const TVector<T>& a, const TVector<T>& b) {
    if (!a || !b) {
        throw(std::invalid_argument("Vector has no Data."));
    }
    if (a.GetSize() != b.GetSize()) {
        throw(std::invalid_argument("Vectors sizes are different. Cannot apply +."));
    }
    T res{};
    size_t size = a.GetSize();
    for (size_t i = 0; i < size; ++i) {
        res += a(i) * b(i);
    }
    return res;
}


template <> 
double TVector<TMatrix<double>>::Norm2(const TVector<TMatrix<double>>& v) = delete;

template <>
double TVector<TMatrix<double>>::InnerProd(const TVector<TMatrix<double>>& v, const TVector<TMatrix<double>>& u) = delete;

template std::ostream& operator <<(std::ostream& out, const TVector<double>& v);
template bool operator ==(const TVector<double>& a, const TVector<double>& b);
template bool operator !=(const TVector<double>& a, const TVector<double>& b);
template TVector<double> operator +(const TVector<double>& a, const TVector<double>& b);
template TVector<double> operator -(const TVector<double>& a, const TVector<double>& b);
template TVector<double> operator *(const TVector<double>& a, const double coeff);
template double InnerProd(const TVector<double>& a, const TVector<double>& b);

template class TVector<double>;
template class TVector<TMatrix<double>>;