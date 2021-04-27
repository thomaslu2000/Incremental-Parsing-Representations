import numpy as np
from nltk.tree import Tree
from benepar import ptb_unescape
from treebanks import ParsingExample


def cb_traverse(tree):
    if isinstance(tree, str):
        return tree
    elif len(tree) == 1:
        if isinstance(tree[0], str):
            return tree
        return cb_traverse(tree[0])
    else:
        left_child = cb_traverse(tree[0])
        if len(tree) == 2:
            right_tree = tree[1]
        else:
            right_tree = Tree("", tree[1:])
        right_child = cb_traverse(right_tree)
        return Tree(tree.label(), [left_child, right_child])


def collapse_binarize(tree):
    c = cb_traverse(tree)
    if len(c) == 1 and isinstance(c[0], str):
        c = Tree(tree.label(), [c])
    return c


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
            right_tree = Tree('', tree[1:])
        right_child = cub_traverse(right_tree)
        return Tree(default_label, [left_child, right_child])


def collapse_unlabel_binarize(tree, default_label='S'):
    c = cub_traverse(tree, default_label=default_label)
    if len(c) == 1 and isinstance(c[0], str):
        c = Tree(default_label, [c])
    return c


def random_subspan(tree):
    if tree is None or isinstance(tree, str) or isinstance(tree[0], str) or np.random.random() < 0.3:
        return tree
    return random_subspan(tree[np.random.choice(len(tree))])


def random_parsing_subspan(p, tf=None):
    tree = p.tree
    t = random_subspan(tree)

    # if subspan too small, just return the whole tree
    if t is not None and not isinstance(tree, str) and len(t.leaves()) >= 3:
        words = ptb_unescape.ptb_unescape(t.leaves())
        sp_after = ptb_unescape.guess_space_after(t.leaves())
        p = ParsingExample(tree=t, words=words, space_after=sp_after)
    if tf is not None and p.tree is not None:
        p.tree = tf(p.tree)
    return p
