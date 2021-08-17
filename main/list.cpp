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

ListNode* merge_list(ListNode* l1, ListNode* l2) {
    ListNode* dummy = new ListNode(0), *node = dummy;
    while ( l1 && l2 ) {
        if ( l1->val <= l2->val ) {
            node->next = l1;
            l1 = l1->next;
        } else {
            node->next = l2;
            l2 = l2->next;
        }
        node = node->next;
    }
    node->next = l1? l1: l2;
    return dummy->next;
}

ListNode* get_list(std::vector<int>&& vec) {
    ListNode* dummy = new ListNode(0), *node = dummy;
    for ( auto n: vec ) {
        node->next = new ListNode(n);
        node = node->next;
    }
    return dummy->next;
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
    auto head3 = get_list({1, 3, 5, 7, 9 });
    auto head4 = merge_list(head3, head2);
    print_list(head4);
}