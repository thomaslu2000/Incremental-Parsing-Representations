import sys
sys.path.append("../src")
import torch
import torch.nn.functional as F
from benepar import decode_chart
from benepar import nkutil
from benepar import parse_chart
from tree_transforms import collapse_unlabel_binarize, collapse_binarize
from treebanks import ParsingExample, Treebank
import numpy as np
from nltk.tree import Tree
from benepar import tetra_tag


class IParser():
    def __init__(self, model_path_base):
        #         self.info = torch.load(model_path_base, map_location=lambda storage, location: storage)
        self.parser = parse_chart.ChartParser.from_trained(model_path_base)

    def remove_pos(self, tree):
        """
        mutates tree to remove the POS that has to be added
        """
        if isinstance(tree, str) or len(tree) < 1:
            return
        for i in range(len(tree)):
            if isinstance(tree[i], str):
                continue
            elif tree[i].label() == 'UNK':
                tree[i] = tree[i][0]
            else:
                self.remove_pos(tree[i])

    def parse_sentence(self, words):
        if isinstance(words, str):
            words = words.split()
        if words[-1] == '.':
            words.pop(-1)
        treebank = Treebank([
            ParsingExample(tree=None, words=words, space_after=[True for w in words],
                           _pos=[(w, 'UNK') for w in words])
        ])
        tree, cats = self.parser.parse(treebank, return_cats=True, tau=0)[0]
        cats = cats.argmax(-1).cpu().numpy().tolist()
        self.remove_pos(tree)
        for i in range(len(tree.leaves())):
            tree[tree.leaf_treeposition(i)] = tree[tree.leaf_treeposition(
                i)] + "\n {}".format(cats[i + 1])
        return tree, cats

    def tree_from_cats(self, cats, words='cat'):
        tree = self.parser.parse([cats], tau=0)[0]
        self.remove_pos(tree)
        return tree

    def tree_to_tag(self, tree):
        tags = [len(self.parser.tetra_tag_system.tag_vocab)] + self.parser.tetra_tag_system.ids_from_tree(
            tree) + [len(self.parser.tetra_tag_system.tag_vocab)]
        mask = [i in self.parser.tetra_leaves for i in tags]
        tags = torch.tensor([tags])
        mask[0] = mask[-1] = True
        padding_mask = [True for i in tags]
        padding_mask = torch.tensor([padding_mask])

        encoder_in = F.one_hot(tags, num_classes=len(
            self.parser.tetra_tag_system.tag_vocab) + 1).float()
        encoder_in = self.parser.back_project(encoder_in)
        encoder_in = self.parser.back_add_timing(
            self.parser.morpho_emb_dropout(encoder_in))

        annotations = self.parser.back_cycle(
            encoder_in, padding_mask)

        logits = self.parser.f_back(annotations)
        return logits[0][mask].argmax(-1).numpy()
