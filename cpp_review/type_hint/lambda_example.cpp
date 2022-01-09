#include <iostream>
#include <vector>

template<class Func>
void call_twice(const Func& func) {
    func(0);
    func(1);
    std::cout << "Fun size: " << sizeof(func) << std::endl;
}

void call_twice_v2(std::function<void(int)> const & func) {
    func(0);
    func(1);
    std::cout << "Fun size: " << sizeof(func) << std::endl;
}

template< class Func>
void fetch_data(const Func& func) {
    for (int i = 0; i < 32; ++i) {
        func(i);
        func(i + 0.1f);
    }
}

int main() {
    int count = 1;
    auto func = [&](int x) {
        count++;
        std::cout << x << " | " << count << std::endl;
    };
    call_twice(func);
    call_twice_v2(func);

    auto func_v2 = [&](auto x) {
        count++;
        std::cout << x << " | " << count << std::endl;
    };
    call_twice(func_v2);

    // cpp20
    auto func_v3 = [&]<class T> (T x) {
        count++;
        std::cout << x << " | " << count << std::endl;
    };
    call_twice(func_v3);

    std::vector<int> int_vec;
    std::vector<float> float_vec;
    fetch_data([&](auto const& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr ( std::is_same_v<T, int> ) {
            int_vec.push_back(x);
        } else if constexpr ( std::is_same_v<T, float> ) {
            float_vec.push_back(x);
        }
    });
    for (auto & x: float_vec ) {
        std::cout << x << " ";
    }
}