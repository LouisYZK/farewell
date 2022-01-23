// reduce sin(x)的加速比实验

#include <iostream>
#include <vector>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <cmath>

#include "ticktock.h"


int main() {
    int n = 1 << 26;
    std::vector<float> a(n);
    std::vector<float> b(n);

    TICK(for);
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::sin(i);
    }
    TOCK(for);

    TICK(para_for);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), 
        [&] (tbb::blocked_range<size_t> r) {
            for ( size_t i = r.begin(); i < r.end(); ++i ) {
                a[i] = std::sin(i);
            }
    });
    TOCK(para_for);

    float res = 0;
    TICK(reduce);
    for ( size_t i = 0; i < n; ++i ) {
        res += a[i];
    }
    TOCK(reduce);
    std::cout << "res1: " << res << std::endl;


    TICK(para_reduce);
    res = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n),
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
    TOCK(para_reduce);
    std::cout << "papra reduce res: " << res << std::endl;
}