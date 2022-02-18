#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
// #include <omp.h>
#include <tbb/parallel_for.h>

// L1: 32KB
// L2: 256KB
// L3: 12MB

constexpr size_t n = 1<<27;  // 512MB

std::vector<float> a(n);

void BM_ordered(benchmark::State &bm) {
    for (auto _: bm) {
// #pragma omp parallel for
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n),
            [&] (tbb::blocked_range<size_t> r) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    benchmark::DoNotOptimize(a[i]);
                }
            }
        );
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ordered);

static uint32_t randomize(uint32_t i) {
	i = (i ^ 61) ^ (i >> 16);
	i *= 9;
	i ^= i << 4;
	i *= 0x27d4eb2d;
	i ^= i >> 15;
    return i;
}

void BM_random_4B(benchmark::State &bm) {
    for (auto _: bm) {
// #pragma omp parallel for
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n),
            [&] (tbb::blocked_range<size_t> r) {
                 for (size_t i = r.begin(); i < r.end(); i++) {
                    size_t r = randomize(i) % n;
                    benchmark::DoNotOptimize(a[r]);
                }
            }
        );
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_random_4B);

void BM_random_64B(benchmark::State &bm) {
    for (auto _: bm) {
// #pragma omp parallel for
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n / 16),
            [&] (tbb::blocked_range<size_t> r) {
                 for (size_t i = r.begin(); i < r.end(); i++) {
                    size_t r = randomize(i) % (n / 16);
                    for (size_t j = 0; j < 16; j++) {
                        benchmark::DoNotOptimize(a[r * 16 + j]);
                    }
                }
            }
        );
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_random_64B);

void BM_random_4KB(benchmark::State &bm) {
    for (auto _: bm) {
// #pragma omp parallel for
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n / 1024),
            [&] (tbb::blocked_range<size_t> r) {
                 for (size_t i = r.begin(); i < r.end(); i++) {
                    size_t r = randomize(i) % (n / 1024);
                    for (size_t j = 0; j < 1024; j++) {
                        benchmark::DoNotOptimize(a[r * 1024 + j]);
                    }
                }
            }
        );
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_random_4KB);

void BM_random_4KB_aligned(benchmark::State &bm) {
    float *a = (float *)_mm_malloc(n * sizeof(float), 4096);
    memset(a, 0, n * sizeof(float));
    for (auto _: bm) {
// #pragma omp parallel for
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n / 1024),
            [&] (tbb::blocked_range<size_t> r) {
                 for (size_t i = r.begin(); i < r.end(); i++) {
                    size_t r = randomize(i) % (n / 1024);
                    for (size_t j = 0; j < 1024; j++) {
                        benchmark::DoNotOptimize(a[r * 1024 + j]);
                    }
                }
            }
        );
        benchmark::DoNotOptimize(a);
    }
    _mm_free(a);
}
BENCHMARK(BM_random_4KB_aligned);

void BM_random_64B_prefetch(benchmark::State &bm) {
    for (auto _: bm) {
// #pragma omp parallel for
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n / 16),
            [&] (tbb::blocked_range<size_t> r) {
                 for (size_t i = r.begin(); i < r.end(); i++) {
                    size_t next_r = randomize(i + 64) % (n / 16);
                    _mm_prefetch(&a[next_r * 16], _MM_HINT_T0);
                    size_t r = randomize(i) % (n / 16);
                    for (size_t j = 0; j < 16; j++) {
                        benchmark::DoNotOptimize(a[r * 16 + j]);
                    }
                }
            }
        );
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_random_64B_prefetch);

BENCHMARK_MAIN();


// -----------------------------------------------------------------
// Benchmark                       Time             CPU   Iterations
// -----------------------------------------------------------------
// BM_ordered               10770727 ns      9785000 ns           64
// BM_random_4B             44554767 ns     39521500 ns           18
// BM_random_64B             5915107 ns      5409913 ns          127  ???? 按说应该比ordered大才对
// BM_random_4KB            13215826 ns     11218873 ns           63
// BM_random_4KB_aligned    12303545 ns     11434554 ns           65
// BM_random_64B_prefetch   22288362 ns     19838526 ns           38