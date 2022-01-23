#include <mutex>
#include <thread>
#include <vector>
#include <iostream>

int main () {
    std::vector<int> arr1; 
    std::vector<int> arr2;
    std::mutex mtx1;

    std::thread t1([&] {
        for (int i = 0; i < 10; ++i ) {
            std::lock_guard grd(mtx1);
            std::cout << "[T1]..." << std::endl;
            arr1.push_back(i);
        }
    });

    std::thread t2([&] {
        for (int i = 0; i < 10; ++i ) {
            std::lock_guard grd(mtx1);
            std::cout << "[T2]..." << std::endl;
            arr1.push_back(i);
        }
    });
    t1.join();
    t2.join();
    std::cout << "Arr size: " << arr1.size() << std::endl;
}