#pragma once
#include "vector.h"
#include "matrix.h"

namespace ChebyshevIterations {

    template <typename T>
    TVector<T> GetParameters(size_t iter_num);

    template <typename T>
    bool SolveSystem(const TMatrix<T>& a, const TVector<T>& b, TVector<T>& result, const size_t iter_num = 32);

} // namespace ChebyshevIterations