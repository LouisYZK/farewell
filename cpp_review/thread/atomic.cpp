#include <thread>
#include<iostream>
#include <atomic>
#include <vector>

int main() {
    // int counter = 0;
    std::atomic<int> counter;
    counter.store(0);
    std::vector<int> vec(2000);
    std::thread t1 ([&] {
        for ( int i = 0; i < 1000; ++i) {
            // counter += 1;
            int index = counter.fetch_add(1);
            vec[index] = i;
        }
    });
    std::thread t2 ([&] {
        for ( int i = 0; i < 1000; ++i ) {
            // counter += 1;
            int index = counter.fetch_add(1);
            vec[index] = i;
        }
    });
    t1.join();
    t2.join();
    std::cout << counter << std::endl;
    std::cout << vec[1000] << std::endl;
}