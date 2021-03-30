import nltk
import numpy as np


## PUBLIC API


class TetraTagSequence(list):
    """A sequence of tetra-tags corresponding to a particular tree."""

    default_ignore_labels = set(["TOP", "ROOT", "VROOT", "S1"])

    @classmethod
    def from_tree(
        cls,
        tree,
        is_right_branch=False,
        ignore_labels=default_ignore_labels,
        right_branching_binarization=True,
    ):
        """Construct a TetraTagSequence from an nltk.Tree object."""
        internal_tags, leaf_tags = tree_to_tags_helper(
            tree, is_right_branch, ignore_labels, right_branching_binarization
        )
        res = [None] * (len(internal_tags) + len(leaf_tags))
        res[::2] = leaf_tags
        res[1::2] = internal_tags
        return cls(res)

    def to_tree(self, leaf_nodes, top_label="TOP"):
        """Convert this sequence of tags to a tree.

        Args:
          leaf_nodes: The leaf nodes for the words in a sentence, which may be
            strings or nltk.Tree objects that each consist of a part-of-speech
            tag and a word.
          top_label: label for the root node of the created tree

        Returns:
          An nltk.Tree object.
        """
        tree_fragment = tags_to_tree_helper(self, leaf_nodes)
        return nltk.Tree(top_label, tree_fragment.convert())


class TetraTagSystem:
    """A tetra-tagging transition system with a fixed tag vocabulary."""

    def __init__(self, tag_vocab=None, trees=None):
        """Constructs a new TetraTagSystem object.

        Args:
          tag_vocab: (optional) A list of all possible tags.
          trees: (optional) An iterable of nltk.Tree objects. If tag_vocab is
            None, a new tag vocabulary will be constructed by iterating these
            trees.
        """
        if tag_vocab is not None:
            self.tag_vocab = tag_vocab
        elif trees is None:
            raise ValueError("Need either tag_vocab or trees")
        else:
            # Construct tag inventory
            tag_vocab = set()
            for tree in trees:
                for tag in TetraTagSequence.from_tree(tree):
                    tag_vocab.add(tag)
            self.tag_vocab = sorted(tag_vocab)

        self.internal_tag_vocab_size = len(
            [tag for tag in self.tag_vocab if tag[0] in "LR"]
        )
        self.leaf_tag_vocab_size = len(
            [tag for tag in self.tag_vocab if tag[0] in "lr"]
        )

        is_leaf_mask = np.concatenate(
            [
                np.zeros(self.internal_tag_vocab_size),
                np.ones(self.leaf_tag_vocab_size),
            ]
        )
        self._internal_tags_only = np.asarray(-1e9 * is_leaf_mask, dtype=float)
        self._leaf_tags_only = np.asarray(
            -1e9 * (1 - is_leaf_mask), dtype=float
        )

        stack_depth_change_by_id = [None] * len(self.tag_vocab)
        for i, tag in enumerate(self.tag_vocab):
            if tag.startswith("l"):
                stack_depth_change_by_id[i] = +1
            elif tag.startswith("R"):
                stack_depth_change_by_id[i] = -1
            else:
                stack_depth_change_by_id[i] = 0
        assert None not in stack_depth_change_by_id
        self._stack_depth_change_by_id = np.array(
            stack_depth_change_by_id, dtype=np.int32
        )

    def tags_from_tree(self, tree):
        """Returns a TetraTagSequence object given an nltk.Tree object."""
        return TetraTagSequence.from_tree(tree)

    def ids_from_tree(self, tree, ignore_unknown=True):
        """Returns a list of label ids given a tree.

        Args:
          tree: An nltk.Tree object
          ignore_unknown: If set to True (the default), syntactic categories
            that can't be represented using the tag vocabulary will be collapsed
            out. If set to False, trees that can't be represented exactly will
            lead to an exception being thrown.

        Returns:
          A list of integer tag ids
        """
        if not ignore_unknown:
            return [
                self.tag_vocab.index(tag) for tag in self.tags_from_tree(tree)
            ]
        return [
            (
                self.tag_vocab.index(tag)
                if tag in self.tag_vocab
                else self.tag_vocab.index(tag[0])
            )
            for tag in self.tags_from_tree(tree)
        ]

    def tree_from_tags(self, tags, leaf_nodes=None, pos=None):
        """Constructs a tree from a tag sequence.

        Args:
          tags: an iterable of tetra-tags
          leaf_nodes: the leaf nodes to use in the constructed tree
          pos: a list of (word, tag) tuples. If leaf_nodes is None, these will
            be used to construct the leaf nodes in the tree.

        Returns:
          An nltk.Tree object.
        """
        if leaf_nodes is None and pos is None:
            raise ValueError("Either leaf_nodes or pos argument is required.")
        elif leaf_nodes is None and pos is not None:
            leaf_nodes = [nltk.Tree(tag, [word]) for word, tag in pos]
        tree = TetraTagSequence(tags).to_tree(leaf_nodes)
        return tree

    def tree_from_ids(self, ids, leaf_nodes=None, pos=None):
        """Constructs a tree from a tag id sequence.

        Args:
          ids: an iterable of integer tag ids
          leaf_nodes: the leaf nodes to use in the constructed tree
          pos: a list of (word, tag) tuples. If leaf_nodes is None, these will
            be used to construct the leaf nodes in the tree.

        Returns:
          An nltk.Tree object.
        """
        tags = [self.tag_vocab[tag_id] for tag_id in ids]
        if len(tags) == 1 and tags[0].startswith("r"):
            tags = ["l" + tags[0][1:]]
        return self.tree_from_tags(tags, leaf_nodes=leaf_nodes, pos=pos)

    def tree_from_logits(self, logits, mask=None, leaf_nodes=None, pos=None):
        """Constructs a tree from a table of logits.

        Args:
          logits: a numpy array of shape (length, tag_vocab_size)
          mask: (optional) a boolean numpy array of shape (length,). Only logits
            corresponding to True entries in the mask will be used. This
            argument may be useful for models that use subword tokenization or
            padding, where not every output location requires a labeling
            decision.
          leaf_nodes: the leaf nodes to use in the constructed tree
          pos: a list of (word, tag) tuples. If leaf_nodes is None, these will
            be used to construct the leaf nodes in the tree.

        Returns:
          An nltk.Tree object.
        """
        tag_ids = self.ids_from_logits(logits, mask)
        return self.tree_from_ids(tag_ids, leaf_nodes=leaf_nodes, pos=pos)

    def ids_from_logits(self, logits, mask=None):
        """Returns the tag ids for the highest-scoring tree given logits.

        Args:
          logits: a numpy array of shape (length, tag_vocab_size)
        mask: (optional) a boolean numpy array of shape (length,). Only logits
          corresponding to True entries in the mask will be used. This
          argument may be useful for models that use subword tokenization or
          padding, where not every output location requires a labeling
          decision.

        Returns:
          A list of integer tag ids
        """
        beam_search = BeamSearch(
            initial_stack_depth=0,
            stack_depth_change_by_id=self._stack_depth_change_by_id,
            max_depth=12,
            keep_per_depth=1,
        )

        last_t = None
        for t in range(logits.shape[0]):
            if mask is not None and not mask[t]:
                continue
            if last_t is not None:
                beam_search.advance(
                    logits[last_t, :] + self._internal_tags_only
                )
            beam_search.advance(logits[t, :] + self._leaf_tags_only)
            last_t = t

        score, best_tag_ids = beam_search.get_path()
        return best_tag_ids


### Internal helpers for converting between trees and tags.


def tree_to_tags_helper(
    tree,
    is_right_branch=False,
    ignore_labels=(),
    right_branching_binarization=True,
):
    """Helper function for converting a tree to a tag sequence."""
    if not isinstance(tree, nltk.Tree):
        tag = f"r" if is_right_branch else f"l"
        return [], [tag]

    sublabels = [tree.label()] if tree.label() not in ignore_labels else []
    while len(tree) == 1 and isinstance(tree[0], nltk.Tree):
        tree = tree[0]
        if tree.label() not in ignore_labels:
            sublabels.append(tree.label())

    if len(tree) == 1 and not isinstance(tree[0], nltk.Tree):
        sublabels = sublabels[:-1]  # Strip POS tag

    if sublabels:
        label = "/" + "/".join(sublabels)
    else:
        label = ""

    if len(tree) == 1:
        child = tree[0]
        assert not isinstance(child, nltk.Tree)
        tag = f"r{label}" if is_right_branch else f"l{label}"
        return [], [tag]

    assert isinstance(tree, nltk.Tree)

    internal_tags = []
    leaf_tags = []
    for i, child in enumerate(tree):
        is_first_child = i == 0
        is_last_child = i == len(tree) - 1
        if right_branching_binarization:
            child_is_right_branch = is_last_child
        else:
            child_is_right_branch = not is_first_child

        child_internal_tags, child_leaf_tags = tree_to_tags_helper(
            child,
            is_right_branch=child_is_right_branch,
            ignore_labels=ignore_labels,
            right_branching_binarization=right_branching_binarization,
        )
        internal_tags += child_internal_tags
        leaf_tags += child_leaf_tags
        if not is_last_child:
            if right_branching_binarization:
                if i == 0:
                    action = f"R{label}" if is_right_branch else f"L{label}"
                else:
                    action = "R"
            else:
                if i == len(tree.children) - 2:
                    action = f"R{label}" if is_right_branch else f"L{label}"
                else:
                    action = "L"
            internal_tags.append(action)

    return internal_tags, leaf_tags


class StackElement:
    pass


class LeafStackElement(StackElement):
    def __init__(self, node):
        self.node = node
        self.label = ()
        self.valency = None

    def set_label(self, label):
        self.label = tuple(label)

    def fill_valency(self, other):
        raise ValueError("No valency to fill!")

    def convert(self):
        node = self.node
        for sublabel in reversed(self.label):
            node = nltk.Tree(sublabel, [node])
        return [node]


class InternalStackElement(StackElement):
    def __init__(self, left_child, right_child):
        assert isinstance(left_child, StackElement)
        assert left_child.valency is None

        self.label = ()

        self.left_child = left_child
        if right_child is None:
            self.right_child_cell = [None]
            self.valency = self.right_child_cell
        elif isinstance(right_child, StackElement):
            self.right_child_cell = [right_child]
            self.valency = right_child.valency
            right_child.valency = None
        else:
            self.right_child_cell = [right_child]
            self.valency = None

    def set_label(self, label):
        self.label = tuple(label)

    def fill_valency(self, other):
        assert self.valency is not None
        self.valency[0] = other
        self.valency = other.valency
        other.valency = None

    def convert(self):
        assert self.valency is None
        assert self.right_child_cell[0] is not None

        res = []
        res += self.left_child.convert()
        res += self.right_child_cell[0].convert()
        for sublabel in reversed(self.label):
            res = [nltk.Tree(sublabel, res)]
        return res


def tags_to_tree_helper(tags, leaf_nodes):
    assert len(tags) % 2 == 1, "Number of actions must be odd"
    assert tags[0].startswith("l")
    assert len(tags) == 1 or tags[-1].startswith("r")
    assert all([(t.startswith("l") or t.startswith("r")) for t in tags[::2]])
    assert all([(t.startswith("L") or t.startswith("R")) for t in tags[1::2]])

    leaf_nodes = [LeafStackElement(leaf) for leaf in leaf_nodes]

    stack = []
    for tag in tags:
        if tag.startswith("l"):
            node = leaf_nodes.pop(0)
            stack.append(node)
        elif tag.startswith("r"):
            node = leaf_nodes.pop(0)
            stack[-1].fill_valency(node)
        elif tag.startswith("L"):
            node = InternalStackElement(stack[-1], None)
            stack[-1] = node
        elif tag.startswith("R"):
            assert len(stack) > 1
            node = InternalStackElement(stack.pop(), None)
            stack[-1].fill_valency(node)
        node.set_label(tag.split("/")[1:])

    assert len(stack) == 1, "Bad final stack size: {}".format(len(stack))
    return stack[0]


### Internal helpers for inference (finding the highest-scoring tag sequence).


class Beam:
    def __init__(self, scores, stack_depths, prev, backptrs, labels):
        self.scores = scores
        self.stack_depths = stack_depths
        self.prev = prev
        self.backptrs = backptrs
        self.labels = labels


class BeamSearch:
    def __init__(
        self,
        initial_stack_depth,
        stack_depth_change_by_id,
        max_depth=12,
        keep_per_depth=1,
        initial_label=None,
    ):
        # Save parameters
        self.stack_depth_change_by_id = stack_depth_change_by_id
        self.valid_depths = np.arange(1, max_depth)
        self.keep_per_depth = keep_per_depth

        # Initialize the beam
        scores = np.zeros(1, dtype=np.float32)
        stack_depths = np.full(1, initial_stack_depth)
        prev = backptrs = labels = None
        if initial_label is not None:
            labels = np.full(1, initial_label)
        self.beam = Beam(scores, stack_depths, prev, backptrs, labels)

    def advance(self, label_logits):
        label_log_probs = label_logits

        all_new_scores = self.beam.scores[:, None] + label_log_probs
        all_new_stack_depths = (
            self.beam.stack_depths[:, None]
            + self.stack_depth_change_by_id[None, :]
        )

        masked_scores = all_new_scores[None, :, :] + np.where(
            all_new_stack_depths[None, :, :]
            == self.valid_depths[:, None, None],
            0.0,
            -np.inf,
        )
        masked_scores = masked_scores.reshape(self.valid_depths.shape[0], -1)
        idxs = np.argsort(-masked_scores)[:, : self.keep_per_depth].flatten()
        backptrs, labels = np.unravel_index(idxs, all_new_scores.shape)

        transition_valid = all_new_stack_depths[
            backptrs, labels
        ] == self.valid_depths.repeat(self.keep_per_depth)

        backptrs = backptrs[transition_valid]
        labels = labels[transition_valid]

        self.beam = Beam(
            all_new_scores[backptrs, labels],
            all_new_stack_depths[backptrs, labels],
            self.beam,
            backptrs,
            labels,
        )

    def get_path(self, idx=0, required_stack_depth=1):
        if required_stack_depth is not None:
            assert self.beam.stack_depths[idx] == required_stack_depth
        score = self.beam.scores[idx]
        assert score > -np.inf

        beam = self.beam
        label_idxs = []
        while beam.prev is not None:
            label_idxs.insert(0, beam.labels[idx])
            idx = beam.backptrs[idx]
            beam = beam.prev

        return score, label_idxs
