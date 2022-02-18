#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "ticktock.h"

float func(int n) {
    static thread_local std::vector<float> tmp;
    for (int i = 0; i < n; i++) {
        tmp.push_back(i / 15 * 2.718f);
    }
    std::reverse(tmp.begin(), tmp.end());
    float ret = tmp[32];
    tmp.clear();
    return ret;
}

float func_ori(int n) {
    std::vector<float> tmp;
    for (int i = 0; i < n; i++) {
        tmp.push_back(i / 15 * 2.71828f);
    }
    std::reverse(tmp.begin(), tmp.end());
    float ret = tmp[32];
    return ret;
}

int main() {
    constexpr int n = 1<<25;

    TICK(first_alloc_ori);
    std::cout << func_ori(n) << std::endl;
    TOCK(first_alloc_ori);

    TICK(second_alloc_ori);
    std::cout << func_ori(n - 1) << std::endl;
    TOCK(second_alloc_ori);

    TICK(first_alloc);
    std::cout << func(n) << std::endl;
    TOCK(first_alloc);

    TICK(second_alloc);
    std::cout << func(n - 1) << std::endl;
    TOCK(second_alloc);

    return 0;
}