#include <variant>
#include <iostream>

void print(std::variant<int, float>& var) {
    if ( std::holds_alternative<int>(var) ) {
        std::cout << std::get<int>(var) << std::endl;
    } else if ( std::holds_alternative<float>(var) ) {
        std::cout << std::get<float>(var) << std::endl;
    }
    std::visit([]( auto const& v) {
        std::cout << "Visit: " << v << std::endl;
    }, var);
}

void add(std::variant<int, float> const & var1,
         std::variant<int, float> const & var2) {
    std::variant<int, float> ret;
    std::visit([&](auto const& v1, auto const& v2) {
        ret = v1 + v2;
        print(ret);
    }, var1, var2);
}

auto add_v2(std::variant<int, float> const & var1,
         std::variant<int, float> const & var2) {
    return std::visit([&](auto const& v1, auto const& v2) 
        -> std::variant<int, float> {
            return v1 + v2;
    }, var1, var2);
}

int main () {
    std::variant<int, float> v = 3;
    print(v);
    v = 3.14f;
    print(v);
    add(v, 10);
    auto ret = add_v2(v, 100);
    print(ret);
}