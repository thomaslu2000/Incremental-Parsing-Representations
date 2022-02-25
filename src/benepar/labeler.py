import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

from . import nkutil
from .partitioned_transformer import (
    ConcatPositionalEncoding,
    FeatureDropout,
    PartitionedTransformerEncoder,
    PartitionedTransformerEncoderLayer,
)
from . import parse_base
from . import retokenization
from . import subbatching
from . import tetra_tag

import nltk


class TreeProcess():
    def __init__(self):
        config = AutoConfig.from_pretrained(
            'kitaev/tetra-tag-en')
        tag_vocab = [config.id2label[i]
                     for i in sorted(config.id2label.keys())]
        self.tetra_tag_system = tetra_tag.TetraTagSystem(tag_vocab=tag_vocab)

        tag_ind = sorted(set([tag[1:]
                              for tag in self.tetra_tag_system.tag_vocab]))
        self.tag_from_ind = tag_ind
        self.tag_ind = {tag: i for i, tag in enumerate(tag_ind)}
        self.tetra_tag_ind = {'': 0, 'L': 1, 'R': 2, 'l': 3, 'r': 4}
        for i, p in enumerate(nltk.load('help/tagsets/upenn_tagset.pickle').keys(), 5):
            self.tetra_tag_ind[p] = i
        self.tetra_tag_ind['#'] = len(self.tetra_tag_ind)

    def simple_tags_and_labels(self, tree, return_tetra=False):
        pos = nltk.pos_tag(tree.leaves())
        pos_iter = iter(pos)
        tags = self.tetra_tag_system.tags_from_tree(tree)
        x = torch.zeros((len(tags), len(self.tetra_tag_ind)))
        y, att, not_leaves, in_tags = [], [], [], []
        for i, tag in enumerate(tags):
            att.append(True)
            if return_tetra:
                in_tags.append(tag[0])
            x[i, self.tetra_tag_ind[tag[0]]] = 1
            leaf = False
            if tag[0] in 'lr':
                _, p = next(pos_iter)
                x[i, self.tetra_tag_ind[p]] = 1
                leaf = True
            not_leaves.append(not leaf)
            y.append(self.tag_ind[tag[1:]])
        ret = {'input': x, 'attention_mask': torch.BoolTensor(att),
               'not_leaves': torch.BoolTensor(not_leaves), 'labels': torch.LongTensor(y)}
        if return_tetra:
            ret['tetra_tags'] = in_tags
            ret['pos'] = pos
        return ret

    def tree_from_preds(self, tetra_tags, labels, pos):
        tags = [t + self.tag_from_ind[label] if t not in 'lr' else t
                for t, label in zip(tetra_tags, labels)]
        try:
            return self.tetra_tag_system.tree_from_tags(tags, pos=pos)
        except:
            print('error making tree', end=' ')
            return nltk.tree.Tree.fromstring("(TOP (S (. .)))")

    def get_tag_label_vocab(self):
        return self.tetra_tag_ind, self.tag_ind


class Labeler(nn.Module):
    def __init__(
        self,
        hparams,
    ):
        super().__init__()
        self.config = locals()
        self.config.pop("self")
        self.config.pop("__class__")
        self.config["hparams"] = hparams.to_dict()

        self.process = TreeProcess()

        self.tag_vocab, self.label_vocab = self.process.get_tag_label_vocab()

        self.d_model = hparams.d_model

        self.ff = nn.Sequential(
            nn.Linear(len(self.tag_vocab), hparams.d_model // 2),
        )

        self.morpho_emb_dropout = FeatureDropout(
            hparams.morpho_emb_dropout)
        self.add_timing = ConcatPositionalEncoding(
            d_model=hparams.d_model,
            max_len=hparams.encoder_max_len,
        )

        encoder_layer = PartitionedTransformerEncoderLayer(
            hparams.d_model,
            n_head=hparams.num_heads,
            d_qkv=hparams.d_kv,
            d_ff=hparams.d_ff,
            ff_dropout=hparams.relu_dropout,
            residual_dropout=hparams.residual_dropout,
            attention_dropout=hparams.attention_dropout,
        )
        encoder_gum = hparams.encoder_gum

        self.encoder = PartitionedTransformerEncoder(
            encoder_layer, hparams.num_layers
        )

        self.f_label = nn.Sequential(
            nn.Linear(hparams.d_model, hparams.d_label_hidden),
            nn.LayerNorm(hparams.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_label_hidden, len(self.label_vocab)),
        )
        self.parallelized_devices = None

    @property
    def device(self):
        if self.parallelized_devices is not None:
            return self.parallelized_devices[0]
        else:
            return next(self.f_label.parameters()).device

    @property
    def output_device(self):
        if self.parallelized_devices is not None:
            return self.parallelized_devices[1]
        else:
            return next(self.f_label.parameters()).device

    def parallelize(self, *args, **kwargs):
        self.parallelized_devices = (torch.device(
            "cuda", 0), torch.device("cuda", 1))
        for child in self.children():
            child.to(self.output_device)

    @classmethod
    def from_trained(cls, model_path, config=None, state_dict=None):
        if model_path is not None:
            data = torch.load(
                model_path, map_location=lambda storage, location: storage
            )
            if config is None:
                config = data["config"]
            if state_dict is None:
                state_dict = data["state_dict"]

        config = config.copy()
        hparams = config["hparams"]

        if "force_root_constituent" not in hparams:
            hparams["force_root_constituent"] = True

        config["hparams"] = nkutil.HParams(**hparams)
        parser = cls(**config)
        parser.load_state_dict(state_dict)
        return parser

    def encode(self, example):
        # x, y = self.process.simple_tags_and_labels(self, tree)
        return example

    def pad_encoded(self, encoded_batch):
        batch = {}
        batch['input'] = nn.utils.rnn.pad_sequence(
            [example["input"] for example in encoded_batch],
            batch_first=True)
        batch['attention_mask'] = nn.utils.rnn.pad_sequence(
            [example["attention_mask"] for example in encoded_batch],
            batch_first=True)
        batch['not_leaves'] = nn.utils.rnn.pad_sequence(
            [example["not_leaves"] for example in encoded_batch],
            batch_first=True)
        batch['labels'] = nn.utils.rnn.pad_sequence(
            [example["labels"] for example in encoded_batch],
            batch_first=True)
        return batch

    def _get_lens(self, encoded_batch):
        return [len(encoded["input"]) for encoded in encoded_batch]

    def encode_and_collate_subbatches(self, examples, subbatch_max_tokens):
        batch_size = len(examples)
        batch_num_tokens = sum(len(x['labels']) for x in examples)
        encoded = [self.encode(example) for example in examples]

        res = []
        for ids, subbatch_encoded in subbatching.split(
            encoded, costs=self._get_lens(encoded), max_cost=subbatch_max_tokens
        ):
            subbatch = self.pad_encoded(subbatch_encoded)
            subbatch["batch_size"] = batch_size
            subbatch["batch_num_tokens"] = batch_num_tokens
            res.append((len(ids), subbatch))
        return res

    def forward(self, batch):
        inp = batch["input"].to(self.device)
        att = batch["attention_mask"].to(self.device)
        x = self.ff(inp)
        encoder_in = self.add_timing(
            self.morpho_emb_dropout(x)
        )
        annotations = self.encoder(encoder_in, att)
        out = self.f_label(annotations)
        return out

    def make_trees(self, data, batch_size=10):
        dev_trees = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                res = self.encode_and_collate_subbatches(
                    data[i:i + batch_size], 4000)
                out = self.forward(res[0][1])
                for j in range(out.shape[0]):
                    dev_trees.append(self.process.tree_from_preds(
                        data[i + j]['tetra_tags'], out[j].argmax(-1), data[i + j]['pos']))
        return dev_trees

    def compute_loss(self, batch):
        out = self.forward(
            batch)

        mask = batch['not_leaves'].to(out.device)
        labels = batch["labels"].to(out.device)

        return F.cross_entropy(
            out[mask],
            labels[mask],
            reduction="sum"
        ) / batch["batch_num_tokens"], (out[mask].argmax(-1) == labels[mask]).cpu().numpy()
