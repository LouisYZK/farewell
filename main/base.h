#include <bits/stdc++.h>
template < class T1, class T2 >
class MyPair {
public:
    MyPair(T1 k, T2 v): _key(k), _val(v) {}
    bool operator < ( const MyPair<T1, T2>& p ) const;
private:
    T1 _key;
    T2 _val;        
};

template <class T1, class T2>
bool MyPair<T1, T2>::operator< (const MyPair<T1, T2>& p ) const {
    return _val < p._val;
}

class Number {
public:
    int myVal;
    static int uniqe;
    Number() {
        myVal = uniqe++;
    }
    Number(const Number& num ) {
        myVal = uniqe++;
    }
};;

int Number::uniqe = 10;
void printNumber(const Number& n) {
    std::cout << "Print Number: " << n.myVal << std::endl;
}