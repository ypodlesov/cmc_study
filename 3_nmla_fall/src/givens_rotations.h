#pragma once

#include "matrix.h"

namespace NGivensRotations {

    namespace {

        template <typename T>
        bool ModifyRows(TMatrix<T>& a, TVector<T>& b, const size_t colIdx, const size_t rowIdx1, const size_t rowIdx2);

        template <typename T>
        bool NullifyColumn(TMatrix<T>& a, TVector<T>& b, const size_t colIdx);
        
    } // namespace

    template <typename T>
    bool SystemToTriangular(TMatrix<T>& a, TVector<T>& b);

    template <typename T>
    bool SolveSystem(TMatrix<T>& a, TVector<T>& b, TVector<T>& x);

} // namespace NGivensRotations
