#include "solve_eq.h"

void Solve(const int64_t n, const std::vector<double>& right_part, std::vector<double>& x) {
    const double A = -1.0 * n * n;
    const double B = 2 * n * n;
    const double MC = 1.0 * n * n;
    std::vector<double> alpha(n, 0.0);
    std::vector<double> betta(n, 0.0);
    std::vector<double> y(n);
    
    y[1] = B;
    alpha[1] = MC / y[1];
    betta[1] = right_part[1] / y[1];

    for(int i = 2; i < n - 1; ++i) {
        y[i] = B + A * alpha[i - 1];
        alpha[i] = MC / y[i];
        betta[i] = (right_part[i] - A * betta[i - 1]) / y[i];
    }

    y[n - 1] = B + A * alpha[n - 2];
    betta[n - 1] = (right_part[n - 1] - A * betta[n - 2]) / y[n - 1];
    x[n - 1] = betta[n - 1];

    for(int i = n - 2; i >= 0; --i) {
        x[i] = alpha[i] * x[i + 1] + betta[i];
    }
}