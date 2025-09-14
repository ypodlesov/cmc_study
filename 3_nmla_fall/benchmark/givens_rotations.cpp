#include <benchmark/benchmark.h>
#include <fstream>
#include <givens_rotations.h>
#include <vector.h>

static void BM_SolveSystem(benchmark::State& state) {
    for (auto _ : state) {
        for (auto _ : state) {
            state.PauseTiming();
            const size_t size = 100;
            TMatrix<double> a(size);
            std::ifstream file("tests/SLAU_var_5.csv");
            for (size_t i = 0; i < size; ++i) {
                for (size_t j = 0; j < size; ++j) {
                    file >> a(i, j);
                    char c;
                    file >> c;
                }
            }
            auto b = TVector<double>::CreateRandom(size);
            TVector<double> x(size);
            x.Nullify();
            state.ResumeTiming();
            NGivensRotations::SolveSystem(a, b, x);
        }
    }
}

BENCHMARK(BM_SolveSystem);

BENCHMARK_MAIN();