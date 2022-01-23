#include <iostream>
#include <vector>
#include <cmath>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <mutex>

#include "ticktock.h"


// 并行筛选、保证顺序、线程安全
int main() {
    size_t n = 1<< 27;
    std::vector<float> a;
    std::mutex mtx;

    a.reserve(n * 2 / 3);
    TICK(filter);
    for ( size_t i = 0; i < n ; ++i) {
        float val = std::sin(i);
        if (val > 0) {
            a.push_back(val);
        }
    }
    TOCK(filter);

    std::vector<float> b;
    b.reserve(n * 2 /3);
    // local thread 变量缓存法，避免临界区竞争锁
    TICK(para_filter);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n),
        [&] (tbb::blocked_range<size_t> r) {
            std::vector<float> local_vec;
            local_vec.reserve(r.size());
            for ( size_t i = r.begin(); i < r.end(); ++i) {
                float val = std::sin(i);
                if ( val > 0 ) {
                    local_vec.push_back(val);
                }
            }
            
            std::lock_guard lck(mtx);
            std::copy(local_vec.begin(), local_vec.end(), std::back_inserter(b));
        } 
    );
    TOCK(para_filter);

    tbb::concurrent_vector<float> c;
    c.reserve(n * 2 /3);
    // 直接用Tbb::并发容器试试看！
    TICK(tbb_filter);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n),
        [&] (tbb::blocked_range<size_t> r ) {
            for ( size_t i = r.begin(); i < r.end(); ++i ) {
                float val = std::sin(i);
                if (val > 0) {
                    c.push_back(val);
                }
            }
        }
    );
    TOCK(tbb_filter);

    std::cout << a.size() << " " << b.size() << " " <<  c.size() << std::endl;

    // filter: 2.75431s
    // para_filter: 0.594113s
    // tbb_filter: 2.14674s
    // 67108862 67108862 67108862
}