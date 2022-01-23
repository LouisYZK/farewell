// tbb 中提供的线程安全的、并发容器

#include <iostream>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <cmath>

int main() {
    size_t n = 1 << 10;
    tbb::concurrent_vector<float> a;

    for ( size_t i = 0; i < n; ++i ) {
        auto it = a.grow_by(2);  // 扩容2 （随机不连续地址）
        *it++ = std::cos(i);
        *it++ = std::sin(i);
    }

    std::cout << a.size() << std::endl;

    tbb::concurrent_vector<float> b(n);
    tbb::parallel_for(
        tbb::blocked_range(b.begin(), b.end()),
        [&] ( tbb::blocked_range<decltype(b.begin())> r) {
            for ( auto it = r.begin(); it != r.end(); ++it) {
                *it += 1.0f;
            }
        }
    );
    std::cout << b[1] << std::endl;
}