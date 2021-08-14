#include <stdc++.h>

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

void merge_sort(std::vector<int>& nums) {
    
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

int main() {
    std::cout << " Hello Bazel! " << std::endl;
    std::vector<int> test_vec_int { 8, 7, 3, 1, 4, 0, 1, 2, 4 };
    std::vector<int> test_vec_int_sort { test_vec_int };
    // std::sort( test_vec_int_sort.begin(), test_vec_int_sort.end() );
    quick_sort(test_vec_int_sort, 0, test_vec_int_sort.size());
    printc(test_vec_int_sort);
    
    std::cout << bin_search(test_vec_int_sort, 3) << std::endl;
    std::cout << bin_search_more(test_vec_int_sort, 4, 1) << std::endl;
}