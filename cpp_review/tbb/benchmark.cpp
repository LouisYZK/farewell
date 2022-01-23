// 使用google性能测试框架；自动迭代多次测试
   
#include <iostream>
#include <vector>
#include <cmath>
#include <benchmark/benchmark.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

constexpr size_t n = 1<<27;
std::vector<float> a(n);

void BM_for(benchmark::State &bm) {
    for (auto _: bm) {
        // fill a with sin(i)
        for (size_t i = 0; i < a.size(); i++) {
            a[i] = std::sin(i);
        }
    }
}
BENCHMARK(BM_for);

void BM_reduce(benchmark::State &bm) {
    for (auto _: bm) {
        // calculate sum of a
        float res = 0;
        for (size_t i = 0; i < a.size(); i++) {
            res += a[i];
        }
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_reduce);

void BM_para_reduce(benchmark::State &bm) {
    for (auto _: bm) {
        float res = 0;
        res = tbb::parallel_reduce(
            tbb::blocked_range<size_t> (0, n),
            (float)0,
            [&] (tbb::blocked_range<size_t> r, float local_res) {
                for ( size_t i = r.begin(); i < r.end(); ++i) {
                    local_res += a[i];
                }
                return local_res;
            },
            [] (float x, float y) {
                return x + y;
            } 
        );
    }
}
BENCHMARK(BM_para_reduce);

BENCHMARK_MAIN();