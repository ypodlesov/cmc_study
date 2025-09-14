#include "givens_rotations.h"
#include "triangular_matrix.h"

#include <cmath>

namespace NGivensRotations {

    namespace {

        template <typename T>
        bool ModifyRows(TMatrix<T>& a, TVector<T>& b, const size_t colIdx, const size_t rowIdx1, const size_t rowIdx2) {
            const size_t aSize = a.GetSize2();
            if (colIdx >= aSize || rowIdx1 >= aSize || rowIdx2 >= aSize) {
                return false;
            }
            const T elem1 = a(rowIdx1, colIdx);
            const T elem2 = a(rowIdx2, colIdx);
            const T denum = std::sqrt(elem1 * elem1 + elem2 * elem2);
            const T c1 = elem1 / denum;
            const T c2 = elem2 / denum;
            for (size_t j = colIdx; j < aSize; ++j) {
                auto tmp = a(rowIdx1, j);
                a(rowIdx1, j) = c1 * a(rowIdx1, j) + c2 * a(rowIdx2, j);
                a(rowIdx2, j) = -(c2 * tmp) + c1 * a(rowIdx2, j);
            }
            b(rowIdx1) = c1 * b(rowIdx1) + c2 * b(rowIdx2);
            b(rowIdx2) = -(c2 * b(rowIdx1)) + c1 * b(rowIdx2);
            return true;
        }

        template <typename T>
        bool NullifyColumn(TMatrix<T>& a, TVector<T>& b, const size_t colIdx) {
            const size_t aSize = a.GetSize1();
            if (aSize != a.GetSize2() || colIdx >= aSize) {
                return false;
            }
            for (size_t i = colIdx + 1; i < a.GetSize1(); ++i) {
                if (!ModifyRows(a, b, colIdx, colIdx, i)) {
                    return false;
                }
            }
            return true;
        }

    } // namespace

    template <typename T>
    bool SystemToTriangular(TMatrix<T>& a, TVector<T>& b) {
        if (a.GetSize1() != a.GetSize2()) {
            return false;
        }
        for (size_t j = 0; j < a.GetSize2() - 1; ++j) {
            if (!NullifyColumn(a, b, j)) {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    bool SolveSystem(TMatrix<T>& a, TVector<T>& b, TVector<T>& x) {
        if (!SystemToTriangular(a, b) || !NTriangularMatrix::SolveSystem(a, b, x)) {
            return false;
        }
        return true;
    }

    template bool SolveSystem(TMatrix<double>& a, TVector<double>& b, TVector<double>& x);

} // namespace NGivensRotations
