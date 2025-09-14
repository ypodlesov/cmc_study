#include <cmath>
#include "chebyshev_iter.h"

namespace ChebyshevIterations {

    template <typename T>
    TVector<T> GetParameters(const TMatrix<T>& a, size_t iter_num) {
        TVector<T> result(iter_num);
        result(0) = 1;
        for (size_t i = 1; i < iter_num; i *= 2) {
            TVector<T> tmp(iter_num);
            for (size_t j = 0; j < 2 * i; ++j) {
                if (j & 1) {
                    tmp(j) = (i << 2) - result((j - 1) >> 1);
                } else {
                    tmp(j) = result(j >> 1);
                }
            }
            result = std::move(tmp);
        }
        const auto [spectrum_min, spectrum_max] = a.GetSpectrumBoundaries();
        std::cout << spectrum_min << ' ' << spectrum_max << std::endl; 
        const auto r_0 = (spectrum_max - spectrum_min) / (spectrum_max + spectrum_min);
        const auto tau_0 = 2 / (spectrum_min + spectrum_max);
        for (size_t i = 0; i < iter_num; ++i) {
            result(i) = -std::cos(result(i) / (iter_num << 1) * M_PI);
            result(i) = tau_0 / (1 + result(i) * r_0);
        }
        return result;
    }

    template <typename T>
    bool SolveSystem(const TMatrix<T>& a, const TVector<T>& b, TVector<T>& result, const size_t iter_num) {
        if (a.GetSize1() != a.GetSize2()) {
            return false;
        }
        auto methodParams = GetParameters(a, iter_num);
        result = TVector<T>(b.GetSize());
        result.Nullify();
        for (size_t iter = 0; iter < iter_num; ++iter) {
            TVector<T> tmp(result.GetSize());
            tmp.Nullify();
            for (size_t i = 0; i < a.GetSize1(); ++i) {
                double v = 0;
                for (size_t j = 0; j < a.GetSize2(); ++j) {
                    v += a(i, j) * result(j);
                }
                tmp(i) = (b(i) - v) * methodParams(iter_num) + result(i);
            }
            result = std::move(tmp);
        }
        return true;
    }

    template bool SolveSystem(const TMatrix<double>& a, const TVector<double>& b, TVector<double>& result, const size_t iterm_num);

} // namespace ChebyshevIterations