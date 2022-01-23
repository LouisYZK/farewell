#include <vector>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <iostream>

class SafeVector {
    std::vector<int> data;
    // mutable std::mutex mtx;
    mutable std::shared_mutex mtx; // 读写锁

    public:
        void push_back(int const& num) {
            // std::lock_guard gd(mtx);
            mtx.lock();
            data.push_back(num);
            mtx.unlock();
        }
        size_t size() const {
            // std::lock_guard gd(mtx);
            mtx.lock_shared();
            size_t ret =  data.size();
            mtx.unlock_shared();
            return ret;
        }
};

int main() {
    SafeVector vec;
    std::thread t1([&] {
        for ( int i = 0; i < 1000; ++i ) {
            vec.push_back(i);
        }
    }) ;
    std::thread t2([&] {
        for ( int i = 0; i < 1000; ++i ) {
            vec.push_back(i);
        }
    });
    t1.join();
    t2.join();
    std::cout << "Vec size: " << vec.size() << std::endl;
}