#include <stdio.h>
#include <stdlib.h>

typedef struct Node
{
    int left, right;
    int sum, add;
    struct Node *child_left, *child_right;
} Node;

Node *
build(int left, int right, int *a)
{ // builds segment tree based on a[left...right-1]
    Node *res = NULL;
    int mid = (left + right) / 2;
    if (left > right) return NULL;
    res = calloc(1, sizeof(Node));
    res->left = left;
    res->right = right;
    res->add = 0;
    if (left == right) {
        res->child_left = NULL;
        res->child_right = NULL;
        res->sum = a[left];
        return res;
    }
    res->child_left = build(left, mid, a);
    res->child_right = build(mid + 1, right, a);
    res->sum = res->child_left->sum + res->child_right->sum;
    return res;
}

int
query_sum(int left, int right, Node *root)
{ // calculate sum of elements a[left...right]
    if (left > root->right || right < root->left) return 0;
    if (left <= root->left && right >= root->right) return root->sum + root->add * (root->right - root->left + 1);
    int tmp = query_sum(left, right, root->child_left);
    tmp += query_sum(left, right, root->child_right);
    if (root->left <= left && root->right >= right) return tmp + root->add * (right - left + 1);
    if (root->right <= right) return tmp + root->add * (root->right - left + 1);
    if (root->left >= left) return tmp + root->add * (right - root->left + 1);
    return 0;
}

void
update(Node *root, int left, int right, int delta)
{ // add delta to a[left...right]
    Node *tmp_l = root->child_left;
    Node *tmp_r = root->child_right;
    if (root->left > right || root->right < left) return;
    if (root->left >= left && root->right <= right) {
        root->add += delta;
        return;
    }
    update(root->child_left, left, right, delta);
    update(root->child_right, left, right, delta);
    root->sum = tmp_l->sum + tmp_l->add * (tmp_l->right - tmp_l->left + 1);
    root->sum += tmp_r->sum + tmp_r->add * (tmp_r->right - tmp_r->left + 1);
}

void
free_tree(Node *root)
{ // free memory devoted to tree
    if (root == NULL) return;
    free_tree(root->child_left);
    free_tree(root->child_right);
    free(root);
}

int main(void)
{
    int n, m;
    if (scanf("%d%d", &n, &m) != 2) {
        fprintf(stderr, "programm: expected 2 integer numbers for input\n");
        exit(1);
    }
    if (n < 0) {
        fprintf(stderr, "programm: expected 1 <= n <= 10000\n");
        exit(1);
    }
    int *a = calloc(n, sizeof(n));
    for (int i = 0; i < n; ++i) {
        a[i] = 0;
    }
    Node *tree = build(0, n-1, a);
    while (m-- > 0) {
        int cmd_type, left, right;
        scanf("%d%d%d", &cmd_type, &left, &right);
        if (cmd_type & 1) {
            int s;
            scanf("%d", &s);
            update(tree, left, right - 1, s);
        } else {
            printf("%d\n", query_sum(left, right - 1, tree));
        }
    }
    free(a);
    free_tree(tree);
    return 0;
}
