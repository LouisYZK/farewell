#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <vector>

#include "ticktock.h"

// 选择合适的粒度; 矩阵转置

int main() {
    size_t n =  1 << 14;
    std::vector<float> a(n * n);
    std::vector<float> b(n * n);
    std::vector<float> c(n * n);
    std::vector<float> d(n * n);

    TICK(transpose);
    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, n, 0, n),
        [&] (tbb::blocked_range2d<size_t> r) {
            for ( size_t i = r.cols().begin(); i < r.cols().end(); ++i ) {
                for ( size_t j = r.rows().begin(); j < r.rows().end(); ++j) {
                    b[i * n + j] = a[j *n + i];
                }
            }
        }
    );
    TOCK(transpose);

    TICK(transpose_g);
    size_t grain = 16;
    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, n, grain, 0, n, grain),
        [&] (tbb::blocked_range2d<size_t> r) {
            for ( size_t i = r.cols().begin(); i < r.cols().end(); ++i ) {
                for ( size_t j = r.rows().begin(); j < r.rows().end(); ++j) {
                    c[i * n + j] = d[j *n + i];
                }
            }
        },
        tbb::simple_partitioner{}
    );
    TOCK(transpose_g);
}