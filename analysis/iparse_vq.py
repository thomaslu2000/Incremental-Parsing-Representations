import sys
sys.path.insert(0, "../src")
from benepar import Parser, InputSentence
from benepar.parse_base import BaseInputExample
import numpy as np
from nltk.tree import Tree
import treebanks
from treebanks import ParsingExample, Treebank
from benepar import tetra_tag
import torch
import torch.nn.functional as F


class IParser():
    def __init__(self, model_path_base):
        self.nltk_wrapper = Parser(model_path_base)
        self.parser = self.nltk_wrapper._parser
        self.tag_vocab = self.parser.tag_vocab

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

        if isinstance(words, BaseInputExample):
            inputs = [words]
            words = words.words
        else:
            inputs = [InputSentence(words=words)]
            inputs = [self.nltk_wrapper._with_missing_fields_filled(
                x) for x in inputs]

        tree, cats = self.parser.parse(
            inputs, return_cats=True, tau=0)[0]
        cats = cats.tolist()
        self.remove_pos(tree)
        leaf_treepositions = tree.treepositions('leaves')
        for i, leaf_treeposition in enumerate(leaf_treepositions):
            tree[leaf_treeposition] = tree[leaf_treeposition
                                           ] + "\n {}".format(cats[i])
        return tree, cats

    def parse_batch(self, batched_words):
        inputs = []
        for words in batched_words:
            if isinstance(words, str):
                words = words.split()

            if isinstance(words, BaseInputExample):
                inputs.append(words)
            else:
                inp = InputSentence(words=words)
                inp = self.nltk_wrapper._with_missing_fields_filled(inp)
                inputs.append(inp)

        predicted = self.parser.parse(
            inputs, return_cats=True, tau=0, subbatch_max_tokens=2000)
        for j, (tree, cats) in enumerate(predicted):
            cats = cats.tolist()
            self.remove_pos(tree)
            leaf_treepositions = tree.treepositions('leaves')
            for i, leaf_treeposition in enumerate(leaf_treepositions):
                tree[leaf_treeposition] = tree[leaf_treeposition
                                               ] + "\n {}".format(cats[i])
            predicted[j] = (tree, cats)

        return predicted

    def tree_from_cats(self, cats, words=None, no_cat=False):
        tree = self.parser.parse([cats], tau=0)[0]
        self.remove_pos(tree)
        if words is not None:
            if isinstance(words, str):
                words = words.split()
            elif isinstance(words, BaseInputExample):
                words = words.words
            leaf_treepositions = tree.treepositions('leaves')
            for i, leaf_treeposition in enumerate(leaf_treepositions):
                if no_cat:
                    tree[leaf_treeposition] = words[i]
                else:
                    tree[leaf_treeposition] = "{} \n".format(words[i]
                                                             ) + tree[leaf_treeposition]
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

    def load_dev(self, path='../data/22.auto.clean', max_len_dev=40):
        dev_treebank = treebanks.load_trees(
            path, None, 'default'
        )
        dev_treebank = dev_treebank.filter_by_length(max_len_dev)
        return dev_treebank

    def load_train(self, path='../data/02-21.10way.clean', max_len_dev=40):
        dev_treebank = treebanks.load_trees(
            path, None, 'default'
        )
        dev_treebank = dev_treebank.filter_by_length(max_len_dev)
        return dev_treebank

    def get_tag_dist(self, dev_treebank):
        dev_predicted_and_cats, encoded = self.parser.parse(
            dev_treebank.without_gold_annotations(),
            subbatch_max_tokens=2000,
            tau=0.0,
            return_cats=True,
            return_encoded=True
        )
        tag_distribution = np.zeros(
            (len(self.tag_vocab), self.parser.d_cats))
        for (dev_tree, cat), example, encode in zip(dev_predicted_and_cats, dev_treebank, encoded):

            # w x
            categories = cat.argmax(-1)
            try:
                for i, pos in enumerate(example.pos()):
                    tag_distribution[self.tag_vocab[pos[1]],
                                     categories[encode['words_from_tokens'][i + 1]]] += 1
            except:
                pass
        return tag_distribution
