#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>


int download(const std::string& file ) {
    for ( int i = 0; i < 10; ++i ) {
        std::cout << "Downloading " << file  
                  << i * 10  << "% ..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
}

int interact() {
    std::string name;
    std::cin >> name;
    std::cout << "Welcome " << name << std::endl;
}

std::vector<std::thread> pool;

class ThreadPool {
    std::vector<std::thread> m_pool;
    
    public:
        void push_thread(std::thread&& t) {
            m_pool.push_back(std::move(t));
        }
        ~ThreadPool() {
            for (auto& t: m_pool) {
                t.join();
            }
        }
};

ThreadPool mpool;

int main() {
    // download("test.zip"); Block
    // std::thread t1([&]{
    //     download("test.zip");
    // });
    {
        std::thread t1([&]{
            download("test.zip");
        });
        // t1.detach(); // 将局部的线程对象脱离std::thread的控制，进程退出后退出
        // pool.push_back(std::move(t1));  // 放入全局pool 延长生命周期；
        mpool.push_thread(std::move(t1));
    }
    interact();
    // t1.join();
}