#include <optional>
#include <iostream>
#include <cmath>

std::optional<float> mysqrt(float x) {
    if ( x > 0.f) {
        return std::sqrt(x);
    } else {
        return std::nullopt;
    }
}

int main() {
    auto ret = mysqrt(1.1f);
    if ( ret.has_value()) {
        std::cout << "Success: " << ret.value() << std::endl;
    } else {
        std::cout << "Failed" << std::endl;
    }

    auto ret2 = mysqrt(-3.14f);
    std::cout << " Opt: " << ret2.value_or(0.0f) << std::endl;
}