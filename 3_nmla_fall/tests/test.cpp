#include <cstddef>
#include <gtest/gtest.h>
#include <matrix.h>
#include <vector.h>
#include <givens_rotations.h>
#include <chebyshev_iter.h>
#include <common.h>
#include <fstream>
#include <filesystem>

TEST(GivensRotations, SolveSystemTest) {
    const size_t size = 100;
    TMatrix<double> a(size);
    std::ifstream file("../tests/SLAU_var_5.csv");
    EXPECT_TRUE(file.is_open());
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            file >> a(i, j);
            char c;
            file >> c;
        }
    }
    EXPECT_TRUE(a.GetSize1() == a.GetSize2() && a.GetSize1() == size);
    auto b = TVector<double>::CreateRandom(size);
    TVector<double> x(b.GetSize());
    x.Nullify();
    EXPECT_TRUE(NGivensRotations::SolveSystem(a, b, x));
    ASSERT_TRUE(TMatrix<double>::IsTriangular(a, ETriangularType::Upper));
    TVector<double> res = a * x;
    ASSERT_TRUE(RoughEq(TVector<double>::Norm2(res - b), 0.0, 1e-6));
}

TEST(ChebyshevIterations, SolveSystemTest) {
    const size_t size = 100;
    TMatrix<double> a(size);
    std::ifstream file("../tests/SLAU_var_5.csv");
    EXPECT_TRUE(file.is_open());
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            file >> a(i, j);
            char c;
            file >> c;
        }
    }
    EXPECT_TRUE(a.GetSize1() == a.GetSize2() && a.GetSize1() == size);
    auto b = TVector<double>::CreateRandom(size);
    TVector<double> x(b.GetSize());
    x.Nullify();
    EXPECT_TRUE(ChebyshevIterations::SolveSystem(a, b, x, 2));
    TVector<double> res = a * x;
    ASSERT_TRUE(RoughEq(TVector<double>::Norm2(res - b), 0.0, 1e-6));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}