#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_sort.h>

#include "ticktock.h"


template <class T>
void quick_sort(T* data, size_t size) {
    if ( size < 1) return ;
    size_t mid = std::hash<size_t>{}(size);
    mid ^= std::hash<void *>{}(static_cast<void *>(data));
    mid %= size;

    std::swap(data[0], data[mid]);
    T pivot = data[0];
    size_t left = 0, right = size - 1;
    while ( left < right) {
        while ( left < right && data[right] >= pivot)
            right--;
        if ( left < right ) 
            data[left++] = data[right];
        while ( left < right && data[left] <= pivot )
            left++;
        if (left < right) 
            data[right--] = data[left];
    }
    data[left] = pivot;

        quick_sort(data, left);
        quick_sort(data + left + 1, size - left - 1);

}

template <class T>
void quick_sort_parallel(T* data, size_t size) {

    if ( size < 1) return ;
    size_t mid = std::hash<size_t>{}(size);
    mid ^= std::hash<void *>{}(static_cast<void *>(data));
    mid %= size;

    std::swap(data[0], data[mid]);
    T pivot = data[0];
    size_t left = 0, right = size - 1;
    while ( left < right) {
        while ( left < right && data[right] >= pivot)
            right--;
        if ( left < right ) 
            data[left++] = data[right];
        while ( left < right && data[left] <= pivot )
            left++;
        if (left < right) 
            data[right--] = data[left];
    }
    data[left] = pivot;
        tbb::parallel_invoke(
            [&] {
                quick_sort_parallel(data, left);
            }, [&] {
                quick_sort_parallel(data + left +1, size - left -1);
            }
        );
}


template <class T>
void quick_sort_parallel_crip(T* data, size_t size) {

    if ( size < 1) return ;
    if ( size < (1 << 16 )){
        std::sort(data, data+size, std::less<T>{});
        return;
    }
    size_t mid = std::hash<size_t>{}(size);
    mid ^= std::hash<void *>{}(static_cast<void *>(data));
    mid %= size;

    std::swap(data[0], data[mid]);
    T pivot = data[0];
    size_t left = 0, right = size - 1;
    while ( left < right) {
        while ( left < right && data[right] >= pivot)
            right--;
        if ( left < right ) 
            data[left++] = data[right];
        while ( left < right && data[left] <= pivot )
            left++;
        if (left < right) 
            data[right--] = data[left];
    }
    data[left] = pivot;
        tbb::parallel_invoke(
            [&] {
                quick_sort_parallel(data, left);
            }, [&] {
                quick_sort_parallel(data + left +1, size - left -1);
            }
        );
}

int main() {
    size_t n = 1 << 24;
    std::vector<int> arr(n);
    std::generate(arr.begin(), arr.end(), std::rand);

    std::vector<int> arr0 {arr};
    TICK(std_sort);
    std::sort<int>(arr0.begin(), arr0.end(), std::less<int>{});
    TOCK(std_sort);

    std::vector<int> arr1 {arr};
    TICK(quick_sort);
    quick_sort(arr1.data(), arr1.size());
    TOCK(quick_sort);

    std::vector<int> arr2 {arr};
    TICK(quick_sort_parallel);
    quick_sort_parallel(arr2.data(), arr2.size());
    TOCK(quick_sort_parallel);

    std::vector<int> arr3 {arr};
    TICK(quick_sort_parallel_crip);
    quick_sort_parallel_crip(arr3.data(), arr3.size());
    TOCK(quick_sort_parallel_crip);

    std::vector<int> arr4 {arr};
    TICK(tbb_sort);
    tbb::parallel_sort(arr4.begin(), arr4.end(), std::less<int>{});
    TOCK(tbb_sort);

    // std_sort: 1.2885s
    // quick_sort: 1.75107s
    // quick_sort_parallel: 1.00226s
    // quick_sort_parallel_crip: 0.962619s
    // tbb_sort: 0.308995s

    // 按理说，经过并行+ crip的优化手写qick_sort可以达到跟tbb_sort一样程度优化，不知为何这次测试结果不对；
    // [NOTE]
    // 第一次手写改进quick_sort, 加了几个参数:
    // void quick_sort(T* data, size_t size, parallel = false, crip = false);
    // 函数体内部想通过if语句来判断是否开启并行优化、crip优化等；但实验发现没有效果；初步推断是内部if逻辑导致编译没有优化
    return 0;
}