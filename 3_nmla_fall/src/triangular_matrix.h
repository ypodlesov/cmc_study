#pragma once

#include "matrix.h"
#include "vector.h"

namespace NTriangularMatrix {

    template <typename T>
    bool SolveSystem(const TMatrix<T>& a, const TVector<T>& b, TVector<T>& x);

} // namespace NTriangularMatrix
