import sys
sys.path.insert(0, "../src")
from benepar import Parser, InputSentence
from benepar.parse_base import BaseInputExample
import numpy as np
from nltk.tree import Tree


class IParser():
    def __init__(self, model_path_base):
        self.nltk_wrapper = Parser(model_path_base)
        self.parser = self.nltk_wrapper._parser

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
            inputs = [self.nltk_wrapper._with_missing_fields_filled(x) for x in inputs]

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

    def tree_from_cats(self, cats, words=None):
        tree = self.parser.parse([cats], tau=0)[0]
        self.remove_pos(tree)
        if words is not None:
            if isinstance(words, str):
                words = words.split()
            elif isinstance(words, BaseInputExample):
                words = words.words
            leaf_treepositions = tree.treepositions('leaves')
            for i, leaf_treeposition in enumerate(leaf_treepositions):
                tree[leaf_treeposition] = "{} \n".format(words[i]
                    ) + tree[leaf_treeposition]
        return tree
