// 生产-消费者模式；使用条件变量;
#include <thread>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <iostream>

template <class T>
class MyQueue {
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<T> data;
public:
    void push(T const& x) {
        std::unique_lock grd{mtx};
        data.push_back(x);
        cv.notify_one();
    }

    void push_many(std::initializer_list<T> vals) {
        std::unique_lock grd{mtx};
        std::copy(vals.begin(), vals.end(),
                    std::back_insert_iterator(data));
        cv.notify_all();
    }

    T pop() {
        std::unique_lock lck{mtx};
        cv.wait(lck, [&] {
            return !data.empty();
        });
        T ret = std::move(data.back());
        data.pop_back();
        return ret;
    }
};

int main() {
    MyQueue<int> foods;
    std::thread t1([&] {
        for ( int i = 0; i < 2; i++) {
            auto ret = foods.pop();
            std::cout << "T1 consume: " << ret << std::endl;
        }
    });
    std::thread t2 ([&]{
        for ( int i = 0; i < 2; i++) {
            auto ret = foods.pop();
            std::cout << "T2 consume: " << ret << std::endl;
        }
    });

    foods.push(123);
    foods.push(456);
    foods.push_many({10000, 100001});
    t1.join();
    t2.join();
}