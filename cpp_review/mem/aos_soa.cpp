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


constexpr size_t n = 1 << 26;  // 512MB;

void BM_aos(benchmark::State &bm) {
    struct MyClass
    {
        float x;
        float y;
        float z;
    };
    std::vector<MyClass> mc(n);
    for ( auto _: bm) {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n),
                [&] (tbb::blocked_range<size_t> r) {
                    for ( size_t i = r.begin(); i < r.end(); ++i ) {
                        mc[i].x = mc[i].x + mc[i].y;
                    }
                }
            );
        benchmark::DoNotOptimize(mc);
    }    
}
BENCHMARK(BM_aos);

void BM_soa(benchmark::State &bm) {
    std::vector<float> mc_x(n);
    std::vector<float> mc_y(n);
    std::vector<float> mc_z(n);

    for ( auto _: bm ) {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n),
            [&]( tbb::blocked_range<size_t> r) {
                for ( int i = r.begin(); i < r.end(); ++i ) {
                    mc_x[i] = mc_x[i] + mc_y[i];
                }
            }
        );
        benchmark::DoNotOptimize(mc_x);
        benchmark::DoNotOptimize(mc_y);
        benchmark::DoNotOptimize(mc_z);
    }
}
BENCHMARK(BM_soa);

void BM_aosoa_16(benchmark::State &bm) {
    struct MyClass
    {
        float x[16];
        float y[16];
        float z[16];
    };
    std::vector<MyClass> mc(n / 16);
    for ( auto _ : bm ) {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n / 16),
            [&] (tbb::blocked_range<size_t> r) {
                for ( int i = r.begin(); i < r.end() ; ++i ) {
                    for ( int j = 0; j < 16; ++j) {
                        mc[i].x[j] = mc[i].x[j] + mc[i].y[j];
                    }
                }
            }
        );
        benchmark::DoNotOptimize(mc);
    }
}
BENCHMARK(BM_aosoa_16);


void BM_aosoa_64(benchmark::State &bm) {
    struct MyClass
    {
        float x[64];
        float y[64];
        float z[64];
    };
    std::vector<MyClass> mc(n / 64);
    for ( auto _ : bm ) {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n / 64),
            [&] (tbb::blocked_range<size_t> r) {
                for ( int i = r.begin(); i < r.end() ; ++i ) {
                    for ( int j = 0; j < 64; ++j) {
                        mc[i].x[j] = mc[i].x[j] + mc[i].y[j];
                    }
                }
            }
        );
        benchmark::DoNotOptimize(mc);
    }
}
BENCHMARK(BM_aosoa_64);

BENCHMARK_MAIN();

// 2022-02-17T18:14:45+08:00
// Running ./build/aos_soa
// Run on (8 X 1400 MHz CPU s)
// CPU Caches:
//   L1 Data 32 KiB (x4)
//   L1 Instruction 32 KiB (x4)
//   L2 Unified 256 KiB (x4)
//   L3 Unified 6144 KiB (x1)
// Load Average: 6.91, 5.09, 4.90
// ------------------------------------------------------
// Benchmark            Time             CPU   Iterations
// ------------------------------------------------------
// BM_aos        72944727 ns     55848643 ns           14
// BM_soa        36543514 ns     28750844 ns           32
// BM_aosoa_16   47758857 ns     32780000 ns           23
// BM_aosoa_64   40994555 ns     34755050 ns           20