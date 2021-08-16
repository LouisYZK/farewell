#include <stdc++.h>
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int val = 0, TreeNode* left = nullptr, TreeNode* right = nullptr)
        : val(val), left(left), right(right) {}
};

void pre_order_traval(TreeNode* root) {
    std::stack<TreeNode*> st;
    if ( root ) { st.push(root); }
    while ( !st.empty() ) {
        auto node = st.top();
        if ( node != nullptr ) {
            st.pop();
            if ( node->right ) {st.push(node->right); }
            if ( node->left ) { st.push(node->left); }
            st.push(node);
            st.push(nullptr);
        } else {
            st.pop();
            std::cout << st.top()->val << " ";
            st.pop();
        }
    }
}

void post_order_traval(TreeNode* root) {
    std::stack<TreeNode*> st;
    if ( root ) { st.push(root); }
    while ( !st.empty() ) {
        auto node = st.top();
        if ( node != nullptr ) {
            st.pop();
            st.push(node);
            st.push(nullptr);
            if ( node->right ) {st.push(node->right); }
            if ( node->left ) { st.push(node->left); }
        } else {
            st.pop();
            std::cout << st.top()->val << " ";
            st.pop();
        }
    }
    std::cout << std::endl;
}

void layer_travel(TreeNode* root) {
    std::queue<TreeNode*> q ;
    q.push(root);
    while (!q.empty()) {
        int size = q.size();
        for (int i = 0; i < size; ++i) {
            auto node = q.front();
            q.pop();
            std::cout << node->val << " ";
            if ( node->left) { q.push(node->left); }
            if ( node->right ) { q.push(node->right); }
        }
    }
    std::cout << std::endl;
}

// test tree
//     8
//   7    3
// 1   4
//    0  1
//   2 4
int main() {
    TreeNode* root = new TreeNode(8);
    root->left = new TreeNode(7);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(4);
    root->left->right->left = new TreeNode(0);
    root->left->right->right = new TreeNode(1);
    root->left->right->left->left = new TreeNode(2);
    root->left->right->left->right = new TreeNode(4);
    post_order_traval(root);
    layer_travel(root);
}