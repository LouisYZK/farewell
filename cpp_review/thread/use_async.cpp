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
    std::future<int> ft = std::async([&] {
        return download("xsxsxs.zip");
    });
    interact();
    // ft.wait(); // 显示等待，不返回结果;
    // std::cout << "Wait Over..." << std::endl;
    while (true) {
        auto status = ft.wait_for(std::chrono::milliseconds(1000));
        if ( status == std::future_status::ready ) {
            std::cout << "Wait For Over... " << std::endl;
            break;
        } else {
            std::cout << "Wait For Continue ----" << std::endl;
        }
    }
    auto ret = ft.get();
    std::cout << "Download Result: " << ret << std::endl;
}