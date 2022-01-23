// 并行映射

#include <iostream>
#include <tbb/task_group.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <vector>
#include <cmath>

int main () {
    size_t n = 1 << 26;
    std::vector<float> a(n);
    
    size_t maxt = 4;
    tbb::task_group tg;
    for (size_t t = 0; t < maxt; t++ ) {
        auto beg = t * n / maxt;
        auto end = std::min(n, (t+1) * n / maxt);
        tg.run([&, beg, end] {
            for ( size_t i = beg; i < end; ++i ) {
                a[i] = std::sin(i);
            }
        });
    }
    tg.wait();

    // void parallel_for(const Range &range, const Body &body) 
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), 
        [&] (tbb::blocked_range<size_t> r) {
            for ( size_t i = r.begin(); i < r.end(); ++i) {
                a[i] = std::sin(i);
            }
        });
    
    // 2d;
    size_t m =  1<<13;
    std::vector<float> b (m*m);
    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, m, 0, m), 
        [&](tbb::blocked_range2d<size_t> r) {
            for (size_t i = r.cols().begin(); i < r.cols().end(); i++) {
                for (size_t j = r.rows().begin(); j < r.rows().end(); j++) {
                    b[i * m + j] = std::sin(i) + std::sin(j);
                }
            }
        });
    return 0;
}