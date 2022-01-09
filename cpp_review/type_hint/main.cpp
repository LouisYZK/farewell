#include <memory>
#include <string>
#include <tuple>
#include <iostream>
#include "type_hint.h"
#include "decltype_example.h"

#define SHOW(T) \
    std::cout << cpp_type_name<T>() << std::endl \

struct C {
    std::unique_ptr<C> m_child;
    C* m_parent;
};

int main () {
    auto parent = std::make_unique<C>();
    auto child = std::make_unique<C>();
    child->m_parent = parent.get();

    parent->m_child = std::move(child);

    parent = nullptr;

    auto t = std::tuple<int, float, std::string>(1, 1.11, std::string{ "Hello" });
    auto [x, y, z] = t;
    std::cout << x << " " << std::get<0>(t) << std::endl;

    SHOW(int);
    int a, *p;
    SHOW(decltype(3.14f + a));
    SHOW(decltype(42));
    SHOW(decltype(&a));
    SHOW(decltype(p[0]));
    SHOW(decltype('a'));

    SHOW(decltype(a));    // int
    SHOW(decltype((a)));  // int &
    std::vector<int> av {3,4,5};
    SHOW(decltype(av));

    std::vector<int> vec1 { 1,2,3,4};
    std::vector<float> vec2 { 0.5f, 0.4f, 1.11f, 1.2f};
    auto ret = add_two_vec<int, float>(vec1, vec2);
    for ( auto& item: ret ) {
        std::cout << item << " ";
    }
    SHOW(decltype(ret));
}