// 访问者模式；
// RAII思想，锁一次获取一次释放

#include <mutex>
#include <thread>
#include <vector>
#include <iostream>

class SafeVector {
    std::vector<int> data;
    std::mutex mtx;
public:
    class Accessor {
        SafeVector& m_vec;
        std::unique_lock<std::mutex> m_guard;
    public:
        Accessor(SafeVector& that) : m_vec{ that }, m_guard{ that.mtx} {}
        void push_back(int const & x) {
            return m_vec.data.push_back(x);            
        }
        size_t size() const {
            return m_vec.data.size();
        }
    };
    Accessor access() {
        // return Accessor{*this};
        return { *this };
    }
};

int main() {
    SafeVector vec;
    std::thread t1([&] {
        for ( int i = 0; i < 1000; ++i ) {
            // vec.push_back(i);
            auto acc = vec.access();
            acc.push_back(i);
        }
    }) ;
    std::thread t2([&] {
        for ( int i = 0; i < 1000; ++i ) {
            auto acc = vec.access();
            acc.push_back(i);
        }
    });
    t1.join();
    t2.join();
    auto acc = vec.access();
    std::cout << "Vec size: " << acc.size() << std::endl;
}