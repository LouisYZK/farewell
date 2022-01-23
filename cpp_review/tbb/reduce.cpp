#include <iostream>
#include <tbb/task_group.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <vector>

#include <cmath>

// reduce_sum(sin(x)); \sum_i sin(x_i)

int main() {
    size_t n = 1<< 26;
    float res = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n),                       // Range
        (float)0,                                               // Value 初始值？
        [&] (tbb::blocked_range<size_t> r, float local_res) {   // Body1; 第一层reduce
            for (size_t i = r.begin(); i < r.end(); ++i) {
                local_res += std::sin(i);
            }
            return local_res;
        },
        [](float x, float y) {                                  // 二节点归并逻辑（想想二叉树）
            return x + y;
        } 
    );
    std::cout << res << std::endl;
}