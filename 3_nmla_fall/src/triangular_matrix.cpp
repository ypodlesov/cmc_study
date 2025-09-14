#include "triangular_matrix.h"

namespace NTriangularMatrix {
    
    template <typename T>
    bool SolveSystem(const TMatrix<T>& a, const TVector<T>& b, TVector<T>& x) {
        if (!(TMatrix<T>::IsTriangular(a, ETriangularType::Upper) || a.GetSize1() != a.GetSize2())) {
            return false;
        }
        size_t aSize = a.GetSize1();
        if (!aSize) {
            return false;
        } 
        {
            size_t cnt = 0;
            for (size_t i = 0; i < aSize; ++i) {
                if (a(i, i) == 0) {
                    return false;
                }
            }
        }
        x.Nullify();
        {
            size_t i = aSize - 1;
            while (true) {
                T sum = 0;
                for (size_t j = i + 1; j < x.GetSize(); ++j) {
                    sum += a(i, j) * x(j);
                }
                x(i) = (b(i) - sum) / a(i, i);
                if (i == 0) {
                    break;
                } else {
                    --i;
                }
            }
        }
        return true;
    }

    template bool SolveSystem(const TMatrix<double>& a, const TVector<double>& b, TVector<double>& x);

} // namespace NTriangularMatrix


