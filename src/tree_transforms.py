import numpy as np
from nltk.tree import Tree


def cub_traverse(tree, default_label='S'):
    if isinstance(tree, str):
        return tree
    elif len(tree) == 1:
        if isinstance(tree[0], str):
            return tree
        return cub_traverse(tree[0])
    else:
        left_child = cub_traverse(tree[0])
        if len(tree) == 2:
            right_tree = tree[1]
        else:
            right_tree = Tree('dummy', tree[1:])
        right_child = cub_traverse(right_tree)
        return Tree(default_label, [left_child, right_child])


def collapse_unlabel_binarize(tree, default_label='S'):
    c = cub_traverse(tree, default_label=default_label)
    if len(c) == 1 and isinstance(c[0], str):
        c = Tree(default_label, [c])
    return c
