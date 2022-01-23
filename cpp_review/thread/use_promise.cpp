#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <future>


int download(const std::string& file ) {
    for ( int i = 0; i < 10; ++i ) {
        std::cout << "Downloading " << file  
                  << i * 10  << "% ..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    return 404;
}

int interact() {
    std::string name;
    std::cin >> name;
    std::cout << "Welcome " << name << std::endl;
}

int main() {
    std::promise<int> prom; // promise 是future的底层实现，自己手动创建线程
    std::thread t([&] {
        auto ret = download("xsxsxs");
        prom.set_value(ret);
    });
    interact();
    std::future<int> ft = prom.get_future();
    auto ret = ft.get();
    std::cout << "Download Result: " << ret << std::endl;
    t.join();
}