#include <bits/stdc++.h>

struct ListNode {
    int val;
    ListNode* next;
    ListNode( int val = 0, ListNode* next = nullptr)
        : val(val), next(next) {}
};

void print_list(ListNode* root) {
    while ( root ) {
        std::cout << root->val << " ";
        root = root->next;
    }
    std::cout << std::endl;
}

ListNode* reverse(ListNode* root) {
    ListNode* prev = nullptr;
    ListNode* tail;
    while (root) {
        tail = root->next;
        root->next = prev;
        prev = root;
        root = tail;
    }
    return prev;
}

ListNode* reverse_curse(ListNode* head, ListNode* prev) {
    if ( !head) return prev;
    ListNode* next = head->next;
    head->next = prev;
    return reverse_curse(next, head);
}

int main() {
    ListNode* root = new ListNode(1);
    ListNode* cur= root;
    for ( int i = 2; i < 10; ++i) {
        cur->next = new ListNode(i);
        cur = cur->next;
    }
    auto head = reverse(root);
    print_list(head);
    auto head2 = reverse_curse(head, nullptr);
    print_list(head2);
}