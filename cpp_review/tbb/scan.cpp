#include <iostream>
#include <tbb/task_group.h>
#include <tbb/parallel_scan.h>
#include <tbb/blocked_range.h>
#include <vector>

#include <cmath>

// scan的场景相较于reduce需要存储中间结果 类似于tf.accumulate_sum(n)
// [1, 2, 3, 4] -> [1, 3, 6, 10]

int main() {
    size_t n = 10;
    std::vector<int> a(n);

    int res = tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, n),
        (float)0,
        [&] (tbb::blocked_range<size_t> r, int local_res, auto is_final) {
            for ( size_t i = r.begin(); i < r.end(); ++i) {
                local_res += (int)i;
                if ( is_final) {
                    a[i] = local_res;
                }
            }
            return local_res;
        },
        [] (int x, int y) {
            return x + y;
        }
    );
    for ( auto x: a)  std::cout << x << std::endl;
    std::cout << res;
}