#include <vector>


template<class T1, class T2>
auto add_two_vec(std::vector<T1>& vec1, std::vector<T2>& vec2){
    using new_T = decltype(T1{} + T2{});
    std::vector<new_T> new_vec;
    for ( size_t i = 0; i < vec1.size(); ++i ) {
        new_vec.push_back(vec1[i] + vec2[i]);
    }
    return new_vec;
}