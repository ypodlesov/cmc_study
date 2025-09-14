#include "solve_eq.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

constexpr double I = 1.0;
// constexpr double EPS0 = 1;
// constexpr double EPS2 = 1e-2;

template <class T1, class T2>
bool RoughEq(T1 lhs, T2 rhs, T2 epsilon) { 
    return std::fabs(lhs - rhs) < epsilon;
}

template <class T1, class T2>
bool RoughLT(T1 lhs, T2 rhs, T2 epsilon) {
    return rhs - lhs >= epsilon;
}

template <class T1, class T2>
bool RoughtLTE(T1 lhs, T2 rhs, T2 epsilon) {
    return rhs - lhs > -epsilon;
}

void GeneratRightPart(int64_t n, std::vector<double>& right_part) {
    assert(right_part.size() == static_cast<size_t>(n));
    const double step = I / n;
    for (int i = 1; i < n; ++i) {
        right_part[i] = 2 / ((step * i + 1) * (step * i + 1) * (step * i + 1));
    }
}

static void Test(const int64_t n) {
    std::vector<double> x(n);
    std::vector<double> right_part(n);
    GeneratRightPart(n, right_part);
    Solve(n, right_part, x);
    double val, l2_norm = 0, c_norm = 0;
    const double h = 1.0 / n;
    for(int64_t i = 1; i < n; ++i) {
        val = std::abs(x[i] - I / (i * h + I));
        l2_norm += val * val;
        c_norm = std::max(c_norm, val * h);
    }
    l2_norm = std::sqrt(l2_norm * h * h) / n;
    std::cout << l2_norm << "\n" << c_norm << std::endl;
}

TEST_CASE("Size 2") {
    Test(2);
}

TEST_CASE("Size 4") {
    Test(4);
}

TEST_CASE("Size 8") {
    Test(8);
}

TEST_CASE("Size 16") {
    Test(16);
}

TEST_CASE("Size 32") {
    Test(32);
}

TEST_CASE("Size 64") {
    Test(64);
}

TEST_CASE("Size 128") {
    Test(128);
}

TEST_CASE("Size 256") {
    Test(256);
}