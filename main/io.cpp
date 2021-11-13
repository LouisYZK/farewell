#include <iostream>
#include <vector>

int main() {
    int n;
    std::cin >> n;
    std::vector<int> vec;
    while ( n >0 ) {
        int l, r;
        std::cin >> l >> r;
        // int res =(1 << 30);
        // std::cout << res << std::endl;
        // for ( int i = l; i <= r-1; ++i ) {
        //     for ( int j = i+1; j <= r; ++j ) {
        //         res = std::min(res, i & j);
        //     }
        // }
        // std::cout << res << std::endl;
        int i = 0;
        int res = 0;
        int count = 0;
        int ll = 0;
        while ( (1 << i) <= r ) {
            if ( ( 1<< i) >= l) {
                ++count;
                ll = ( 1<< i);
            }
            ++i;
        }
        if ( count >= 2) {
            res = 0;
        } else if ( count == 1 ) {
            res = ll;
        } else {
            res = l & r;
        }
        vec.push_back(res);
        --n;
    }
    for ( auto& r: vec) {
        std::cout << r << std::endl;
    }
}