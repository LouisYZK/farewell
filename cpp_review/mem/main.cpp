#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <tbb/parallel_for.h>
#include <x86intrin.h>


constexpr size_t n = 1 << 28;

std::vector<float> a(n); //1GB

void BM_serial_add(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < n; i++) {
            a[i] = a[i] + 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_serial_add);

void BM_parallel_add(benchmark::State &bm) {
    for (auto _: bm) {
// #pragma omp parallel for
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n),
            [&] (tbb::blocked_range<size_t> r) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    a[i] = a[i] + 1;
                }       
            }
        );
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_parallel_add);

BENCHMARK_MAIN();