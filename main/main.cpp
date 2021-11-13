#include <bits/stdc++.h>
#include "base.h"

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int val = 0, ListNode* next = nullptr ) : val(val), next(next) {}
};

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int val = 0, TreeNode* left = nullptr, TreeNode* right = nullptr) {}
};

template< typename C >
void printc(const C& data) {
    for ( auto it = std::begin(data); it != end(data); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}

void quick_sort(std::vector<int>& nums, int start, int end) {
    int head = start;
    int tail = end-1;
    if ( head > tail ) { return; }
    int pivot = nums[head];
    while ( head < tail ) {
        while ( head < tail && pivot <= nums[tail] ) {
            tail--;
        } 
        nums[head] = nums[tail];
        while ( head < tail && pivot >= nums[head] ) {
            head++;
        }
        nums[tail] = nums[head];
    }
    nums[head] = pivot;
    quick_sort(nums, start, head);
    quick_sort(nums, head + 1, end);
}


void merge_sort(std::vector<int>& nums, int l, int r, std::vector<int>& temp) {
    if ( l + 1 >= r ) return;
    int mid = l + ( r - l ) / 2;
    merge_sort(nums, l, mid, temp);
    merge_sort(nums, mid, r, temp);
    int p = l, q = mid, cur = l;
    while ( p < mid || q < r ) {
        if ( q == r || (p < mid && nums[p] <= nums[q]) ) {
            temp[cur] = nums[p];
            ++p;
        } else {
            temp[cur] = nums[q];
            ++q;
        }
        ++cur;
    }
    for ( int i = l; i < r; ++i) {
        nums[i] = temp[i];
    }
}

int bin_search(std::vector<int>& nums, int target) {
    int left = 0;
    int right = nums.size();
    while ( left < right ) {
        int mid = left + ( right - left) / 2;
        if ( nums[mid] == target ) {
            return mid;
        } else if ( nums[mid] < target ) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

int bin_search_more(std::vector<int>& nums, int target, int pos = 1) {
    int left = 0;
    int right = nums.size();
    while ( left < right ) {
        int mid = 0;
        if ( pos == 1) 
            mid = left + ( right - left) / 2;
        else 
            mid = left + ( right - left + 1) / 2;
        if ( nums[mid] == target ) {
            if ( pos == 1 ) 
                right = mid;
            else 
                left = mid;
        } else if ( nums[mid] < target ) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    if ( nums[left] == target ) {
        return left;
    }
    return -1;
}

int quick_selection(std::vector<int>& vec, int l, int r) {
    int i = l + 1, j = r;
    while ( i < j ) {
        while ( i < r && vec[i] <= vec[l] ) {
            ++i;
        }
        while ( j > l && vec[j] >= vec[l] ) {
            --j;
        }
        if ( i >= j ) break;
        std::swap(vec[i], vec[j]);
    }
    std::swap(vec[l], vec[j]);
    return j;
}

int find_k_largest(std::vector<int>& vec, int k) {
    int l = 0;
    int r = vec.size() - 1, target = vec.size() - k;
    while ( l < r ) {
        int mid = quick_selection(vec, l, r);
        if ( mid == target) {
            return vec[mid];
        } else if ( mid < target ) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return vec[l];
}



int main() {
    std::cout << " Hello Blade! " << std::endl;
    std::vector<int> test_vec_int { 8, 7, 3, 1, 4, 0, 1, 2, 4 };
    std::vector<int> test_vec_int_sort { test_vec_int };
    // std::sort( test_vec_int_sort.begin(), test_vec_int_sort.end() );
    // quick_sort(test_vec_int_sort, 0, test_vec_int_sort.size());
    std::cout << "Find largest: " << find_k_largest(test_vec_int, 2) << std::endl;
    std::vector<int> temp(test_vec_int_sort.size());
    merge_sort(test_vec_int_sort, 0, test_vec_int_sort.size(), temp);
    printc(test_vec_int_sort);
    
    std::cout << bin_search(test_vec_int_sort, 3) << std::endl;
    std::cout << bin_search_more(test_vec_int_sort, 4, 1) << std::endl;

    MyPair<std::string, int> Astudent { "Jack", 20};
    MyPair<std::string, int> Bstudent { "Rose", 21 };
    std::cout << (Astudent < Bstudent) << std::endl;

    Number a, b = a, c = b;
    printNumber(a); printNumber(b); printNumber(c);
}