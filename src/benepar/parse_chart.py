import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from . import char_lstm
from . import decode_chart
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


class ChartParser(nn.Module, parse_base.BaseParser):
    def __init__(
        self,
        tag_vocab,
        label_vocab,
        char_vocab,
        hparams,
    ):
        super().__init__()
        self.config = locals()
        self.config.pop("self")
        self.config.pop("__class__")
        self.config["hparams"] = hparams.to_dict()

        self.tag_vocab = tag_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab

        self.d_model = hparams.d_model
        try:
            self.tree_transform = hparams.tree_transform
        except:
            self.tree_transform = None
        try:
            self.pretrained_divide = hparams.pretrained_divide
        except:
            self.pretrained_divide = 1.0

        try:
            self.two_label_subspan = hparams.two_label_subspan
        except:
            self.two_label_subspan = None

        self.d_cats = hparams.discrete_cats
        self.use_vq = hparams.use_vq

        try:
            self.mask = hparams.mask
        except:
            self.set_mask(tuple(False for _ in range(self.d_cats)))

        self.back_cycle = None

        self.char_encoder = None
        self.pretrained_model = None
        if hparams.use_chars_lstm:
            assert (
                not hparams.use_pretrained
            ), "use_chars_lstm and use_pretrained are mutually exclusive"
            self.retokenizer = char_lstm.RetokenizerForCharLSTM(
                self.char_vocab)
            self.char_encoder = char_lstm.CharacterLSTM(
                max(self.char_vocab.values()) + 1,
                hparams.d_char_emb,
                hparams.d_model // 2,  # Half-size to leave room for
                # partitioned positional encoding
                char_dropout=hparams.char_lstm_input_dropout,
            )
        elif hparams.use_pretrained:
            self.retokenizer = retokenization.Retokenizer(
                hparams.pretrained_model, retain_start_stop=True
            )
            self.pretrained_model = AutoModel.from_pretrained(
                hparams.pretrained_model)
            self.use_forced_lm = hparams.use_forced_lm
            d_pretrained = self.pretrained_model.config.hidden_size

            if hparams.use_encoder:
                if self.d_cats > 0 and self.use_vq:
                    self.project_pretrained = nn.Linear(
                        d_pretrained, hparams.d_model // 2, bias=False
                    )
                    from vector_quantize_pytorch import VectorQuantize
                    self.vq = VectorQuantize(
                        dim=hparams.d_model // 2,
                        n_embed=self.d_cats,
                        decay=hparams.vq_decay,
                        commitment=hparams.vq_commitment,
                    )
                    self.commit_loss_accum = 0.0
                elif self.d_cats > 0:
                    self.project_in = nn.Linear(
                        d_pretrained, self.d_cats, bias=False)

                    # scaling to adjust output of gpt2
                    self.project_in.weight.data *= 1e-3

                    self.project_out = nn.Linear(
                        self.d_cats, hparams.d_model // 2, bias=False)
                else:
                    self.project_pretrained = nn.Linear(
                        d_pretrained, hparams.d_model // 2, bias=False
                    )
            else:
                self.project_pretrained = nn.Linear(
                    d_pretrained, hparams.d_model, bias=False
                )

        if hparams.use_encoder:
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
            try:
                encoder_gum = hparams.encoder_gum
            except AttributeError:
                encoder_gum = False
            self.encoder = PartitionedTransformerEncoder(
                encoder_layer, hparams.num_layers, gum=encoder_gum
            )
            try:
                if hparams.back_cycle:
                    self.back_add_timing = ConcatPositionalEncoding(
                        d_model=hparams.d_model,
                        max_len=hparams.encoder_max_len,
                    )
                    self.back_use_gold_trees = hparams.back_use_gold_trees
                    self.back_cycle = PartitionedTransformerEncoder(
                        encoder_layer, hparams.back_layers, gum=False
                    )
                    self.tetra_tag_system = tetra_tag.TetraTagSystem(
                        tag_vocab=['L/S', 'R/S', 'l', 'r'])
                    self.tetra_leaves = [2, 3]
                    self.back_project = nn.Linear(
                        len(self.tetra_tag_system.tag_vocab) + 1, hparams.d_model // 2, bias=False
                    )
                    self.back_loss_constant = hparams.back_loss_constant
                    if self.d_cats > 0:
                        seq_dim = self.d_cats
                    else:
                        seq_dim = hparams.d_model // 2
                    self.f_back = nn.Sequential(
                        # layer norm then linear, no relu
                        nn.Linear(hparams.d_model, hparams.d_label_hidden),
                        nn.LayerNorm(hparams.d_label_hidden),
                        nn.ReLU(),
                        nn.Linear(hparams.d_label_hidden, seq_dim),
                    )
                    if hparams.back_loss_type == 'ce':
                        from .loss_functions import cel
                        self.back_criterion = cel
                    elif hparams.back_loss_type == 'tvd':
                        from .loss_functions import tvd
                        self.back_criterion = tvd
                    elif hparams.back_loss_type == 'kl':
                        from .loss_functions import kl
                        self.back_criterion = kl(
                            reduction='sum')
                    elif hparams.back_loss_type == 'js':
                        from .loss_functions import js_gen
                        self.back_criterion = js_gen(
                            reduction='sum')
                    elif hparams.back_loss_type == 'emd':
                        from .loss_functions import emd
                        self.back_criterion = emd
                    elif hparams.back_loss_type == 'mse':
                        self.back_criterion = nn.MSELoss(reduction='sum')
                    else:
                        raise ValueError(
                            "Invalid type of loss for backwards cycle")
                    #  measures of simularities between probability distributions (wasserstein, earth movers, section 2 of wasserstein gan paper)
                    # loss from distribution difference (softmax can end up w inf)

            except:
                self.back_cycle = None
        else:
            self.morpho_emb_dropout = None
            self.add_timing = None
            self.encoder = None

        self.f_label = nn.Sequential(
            nn.Linear(hparams.d_model, hparams.d_label_hidden),
            nn.LayerNorm(hparams.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_label_hidden, max(label_vocab.values())),
        )

        if hparams.predict_tags:
            self.f_tag = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_tag_hidden),
                nn.LayerNorm(hparams.d_tag_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_tag_hidden, max(tag_vocab.values()) + 1),
            )
            self.tag_loss_scale = hparams.tag_loss_scale
            self.tag_from_index = {i: label for label, i in tag_vocab.items()}
        else:
            self.f_tag = None
            self.tag_from_index = None

        self.decoder = decode_chart.ChartDecoder(
            label_vocab=self.label_vocab,
            force_root_constituent=hparams.force_root_constituent,
        )
        self.criterion = decode_chart.SpanClassificationMarginLoss(
            reduction="sum", force_root_constituent=hparams.force_root_constituent
        )

        self.parallelized_devices = None

    def set_mask(self, m):
        assert len(
            m) == self.d_cats, "Length of given mask does not match number of dimensions"
        self.mask = tuple(b for b in m)

        # nn param instead of this
        self.config["hparams"]['mask'] = self.mask

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
            if child != self.pretrained_model:
                child.to(self.output_device)
        self.pretrained_model.parallelize(*args, **kwargs)

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

        if self.two_label_subspan is not None and self.two_label_subspan is not False:
            example = self.two_label_subspan(example, self.tree_transform)

        if self.char_encoder is not None:
            encoded = self.retokenizer(example.words, return_tensors="np")
        else:
            encoded = self.retokenizer(example.words, example.space_after)

        if example.tree is not None:
            encoded["span_labels"] = torch.tensor(
                self.decoder.chart_from_tree(example.tree)
            )
            if self.f_tag is not None:
                encoded["tag_labels"] = torch.tensor(
                    [-100] + [self.tag_vocab[tag]
                              for _, tag in example.pos()] + [-100]
                )
            if self.back_cycle is not None and self.back_use_gold_trees:
                tags = [len(self.tetra_tag_system.tag_vocab)] + self.tetra_tag_system.ids_from_tree(
                    example.tree) + [len(self.tetra_tag_system.tag_vocab)]
                encoded['tetra_tags'] = torch.tensor(tags)
                mask = [i in self.tetra_leaves for i in tags]
                mask[0] = mask[-1] = True
                encoded['tetra_tags_mask'] = torch.tensor(mask)
                padding_mask = [True for i in tags]
                encoded['tetra_padding_mask'] = torch.tensor(padding_mask)
        return encoded

    def pad_encoded(self, encoded_batch):
        batch = self.retokenizer.pad(
            [
                {
                    k: v
                    for k, v in example.items()
                    if (k not in ["span_labels", "tag_labels", "tetra_tags", "tetra_tags_mask", "tetra_padding_mask"])
                }
                for example in encoded_batch
            ],
            return_tensors="pt",
        )
        if encoded_batch and "span_labels" in encoded_batch[0]:
            batch["span_labels"] = decode_chart.pad_charts(
                [example["span_labels"] for example in encoded_batch]
            )
        if encoded_batch and "tag_labels" in encoded_batch[0]:
            batch["tag_labels"] = nn.utils.rnn.pad_sequence(
                [example["tag_labels"] for example in encoded_batch],
                batch_first=True,
                padding_value=-100,
            )
        if encoded_batch and "tetra_tags" in encoded_batch[0]:
            batch["tetra_tags"] = nn.utils.rnn.pad_sequence(
                [example["tetra_tags"] for example in encoded_batch],
                batch_first=True,
                padding_value=len(self.tetra_tag_system.tag_vocab)
            )
        if encoded_batch and "tetra_padding_mask" in encoded_batch[0]:
            batch["tetra_padding_mask"] = nn.utils.rnn.pad_sequence(
                [example["tetra_padding_mask"] for example in encoded_batch],
                batch_first=True
            )
        if encoded_batch and "tetra_tags_mask" in encoded_batch[0]:
            batch["tetra_tags_mask"] = nn.utils.rnn.pad_sequence(
                [example["tetra_tags_mask"] for example in encoded_batch],
                batch_first=True
            )
        return batch

    def _get_lens(self, encoded_batch):
        if self.pretrained_model is not None:
            return [len(encoded["input_ids"]) for encoded in encoded_batch]
        return [len(encoded["valid_token_mask"]) for encoded in encoded_batch]

    def encode_and_collate_subbatches(self, examples, subbatch_max_tokens):
        batch_size = len(examples)
        batch_num_tokens = sum(len(x.words) for x in examples)
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

    def forward(self, batch, tau=1.0, return_cat_logits=False, force_cats=None):
        category_ret = None
        commit_loss = None
        if force_cats is None:
            # Begin calculating from batch
            valid_token_mask = batch["valid_token_mask"].to(self.output_device)

            if (
                self.encoder is not None
                and valid_token_mask.shape[1] > self.add_timing.timing_table.shape[0]
            ):
                raise ValueError(
                    "Sentence of length {} exceeds the maximum supported length of "
                    "{}".format(
                        valid_token_mask.shape[1] - 2,
                        self.add_timing.timing_table.shape[0] - 2,
                    )
                )

            if self.char_encoder is not None:
                assert isinstance(self.char_encoder, char_lstm.CharacterLSTM)
                char_ids = batch["char_ids"].to(self.device)
                extra_content_annotations = self.char_encoder(
                    char_ids, valid_token_mask)
            elif self.pretrained_model is not None:
                input_ids = batch["input_ids"].to(self.device)
                words_from_tokens = batch["words_from_tokens"].to(
                    self.output_device)
                pretrained_attention_mask = batch["attention_mask"].to(
                    self.device)

                extra_kwargs = {}
                if "token_type_ids" in batch:
                    extra_kwargs["token_type_ids"] = batch["token_type_ids"].to(
                        self.device)
                if "decoder_input_ids" in batch:
                    extra_kwargs["decoder_input_ids"] = batch["decoder_input_ids"].to(
                        self.device
                    )
                    extra_kwargs["decoder_attention_mask"] = batch[
                        "decoder_attention_mask"
                    ].to(self.device)
                if self.retokenizer.is_t5 and self.use_forced_lm:
                    pretrained_out = self.pretrained_model(
                        input_ids=input_ids[:, :1],
                        attention_mask=pretrained_attention_mask[:, :1],
                        return_dict=True,
                        **extra_kwargs
                    )
                else:
                    pretrained_out = self.pretrained_model(
                        input_ids, attention_mask=pretrained_attention_mask, return_dict=True, **extra_kwargs
                    )
                features = pretrained_out.last_hidden_state.to(
                    self.output_device) / self.pretrained_divide
                features = features[
                    torch.arange(features.shape[0])[:, None],
                    # Note that words_from_tokens uses index -100 for invalid positions
                    F.relu(words_from_tokens),
                ]
                features.masked_fill_(~valid_token_mask[:, :, None], 0)
                if self.encoder is not None:
                    if self.d_cats > 0 and self.use_vq:
                        # Create a mask that excludes start and stop tokens.
                        # This means that discrete category decisions are made
                        # for each word in the sentence, but not for any special
                        # token. The encoder transformer layers have start/stop
                        # tokens, but those will receive an all-zero content
                        # embedding. Note that GPT-2 does not use a start token,
                        # so the default setting in retokenization.py is to use
                        # the GPT-2 *stop token* embedding at that position. 
                        # That would violate incrementality, so we mask it out.
                        quantization_mask = valid_token_mask.clone()
                        quantization_mask[:, 0] = False
                        quantization_mask[
                            torch.arange(features.shape[0]),
                            valid_token_mask.sum(1) - 1,
                        ] = False

                        projected_features = self.project_pretrained(features)
                        unquantized_features = projected_features[quantization_mask]

                        (quantized_features, categories, commit_loss) = self.vq(
                            unquantized_features)

                        if tau > 0.0:
                            assert self.training, "expected no annealing during eval"
                            quantized_features = (
                                (1.0 - tau) * quantized_features
                                + tau * unquantized_features)

                        extra_content_annotations = torch.zeros_like(projected_features)
                        extra_content_annotations[quantization_mask] = quantized_features

                        category_ret = torch.zeros(
                            projected_features.shape[:-1],
                            dtype=categories.dtype,
                            device=categories.device,
                        )
                        category_ret[quantization_mask] = categories

                        if tau > 0.0:
                            assert self.training, "expected no annealing during eval"
                            extra_content_annotations[:, 0] = tau * projected_features[:, 0]
                            extra_content_annotations[
                                torch.arange(features.shape[0]),
                                valid_token_mask.sum(1) - 1,
                            ] = tau * projected_features[
                                torch.arange(features.shape[0]),
                                valid_token_mask.sum(1) - 1,
                            ]

                            commit_loss = commit_loss * (1.0 - tau)

                        commit_loss = commit_loss * unquantized_features.shape[0] * unquantized_features.shape[1]
                    elif self.d_cats > 0:
                        category_logits = self.project_in(features)

                        mask = torch.BoolTensor(
                            np.tile(self.mask, (category_logits.shape[0], category_logits.shape[1], 1))).to(self.device)
                        category_logits = category_logits.masked_fill(
                            mask, -1e9)
                        if tau > 0:
                            # take the gumbel softmax to
                            # hard = false for training?
                            cats = F.gumbel_softmax(
                                category_logits, tau=tau, hard=True)
                        else:
                            # when tau == 0, take the argmax (for deterministic testing)
                            max_idx = torch.argmax(category_logits, dim=-1)
                            cats = F.one_hot(max_idx, num_classes=self.d_cats).type(
                                torch.FloatTensor).to(self.output_device)
                        if return_cat_logits:
                            category_ret = category_logits
                        else:
                            category_ret = cats

                        extra_content_annotations = self.project_out(cats)
                    else:
                        extra_content_annotations = self.project_pretrained(
                            features)
                        category_ret = extra_content_annotations
            # end calculating from batch
        else:
            # Begin forcing gumbel categories
            assert self.encoder is not None and self.d_cats > 0, "Forcing categories only supported with discretization"
            assert not self.use_vq, "Not implemented with vector quantization"
            categories = force_cats
            cats = []
            valid_token_mask = []
            for i, sen in enumerate(force_cats):
                set_i = []
                for j, cat in enumerate(sen):
                    onehot = np.zeros(self.d_cats)
                    onehot[cat] = 1
                    set_i.append(onehot)
                cats.append(set_i)
                valid_token_mask.append([True for _ in sen])
            cats = torch.Tensor(cats).to(self.output_device)
            valid_token_mask = torch.BoolTensor(
                valid_token_mask).to(self.output_device)
            extra_content_annotations = self.project_out(cats)

        if self.encoder is not None:
            if self.d_cats > 0 and not self.use_vq and tau == 0:
                # disabling dropout to get deterministic results for analysis
                # TODO is this step needed?
                encoder_in = self.add_timing(extra_content_annotations)
            else:
                encoder_in = self.add_timing(
                    self.morpho_emb_dropout(extra_content_annotations)
                )

            annotations = self.encoder(encoder_in, valid_token_mask)
            # Rearrange the annotations to ensure that the transition to
            # fenceposts captures an even split between position and content.
            # TODO(nikita): try alternatives, such as omitting position entirely
            annotations = torch.cat(
                [
                    annotations[..., 0::2],
                    annotations[..., 1::2],
                ],
                -1,
            )
        else:
            assert self.pretrained_model is not None
            annotations = self.project_pretrained(features)

        if self.f_tag is not None:
            tag_scores = self.f_tag(annotations)
        else:
            tag_scores = None

        fencepost_annotations = torch.cat(
            [
                annotations[:, :-1, : self.d_model // 2],
                annotations[:, 1:, self.d_model // 2:],
            ],
            -1,
        )

        # Note that the bias added to the final layer norm is useless because
        # this subtraction gets rid of it
        span_features = (
            torch.unsqueeze(fencepost_annotations, 1)
            - torch.unsqueeze(fencepost_annotations, 2)
        )[:, :-1, 1:]
        span_scores = self.f_label(span_features)
        span_scores = torch.cat(
            [span_scores.new_zeros(
                span_scores.shape[:-1] + (1,)), span_scores], -1
        )
        return span_scores, tag_scores, category_ret, commit_loss

    def split_tag(self, original_idx, copy_idx):

        # give this tag the same probability of being chosen as original
        self.project_in.weight.data[copy_idx] = self.project_in.weight.data[original_idx]
        # add noise ?

        # give this tag the same features for the end transformer
        self.project_out.weight.data[:,
                                     copy_idx] = self.project_out.weight.data[:, original_idx]

    def compute_loss(self, batch, tau=1.0):
        span_scores, tag_scores, cats, commit_loss = self.forward(
            batch, tau, return_cat_logits=True)

        span_labels = batch["span_labels"].to(span_scores.device)
        span_loss = self.criterion(span_scores, span_labels)

        span_loss = span_loss / batch["batch_size"]

        if self.use_vq:
            commit_loss = commit_loss / batch["batch_num_tokens"]
        else:
            commit_loss = 0.0

        if self.back_cycle:
            # print('span loss: ', span_loss.data.cpu().numpy(), end=' | ')
            if self.back_use_gold_trees:
                if self.d_cats > 0:
                    back_loss = self.tetra_to_tags(batch, cats)
                else:
                    back_loss = self.tetra_to_annotations(batch, cats)
            else:
                back_loss = self.predicted_tetra_to_tags(
                    batch, cats, span_scores)

            span_loss += back_loss / batch["batch_num_tokens"]

        if self.use_vq:
            self.commit_loss_accum += float(commit_loss.cpu())

        if tag_scores is None:
            return span_loss + commit_loss
        else:
            tag_labels = batch["tag_labels"].to(tag_scores.device)
            tag_loss = self.tag_loss_scale * F.cross_entropy(
                tag_scores.reshape((-1, tag_scores.shape[-1])),
                tag_labels.reshape((-1,)),
                reduction="sum",
                ignore_index=-100,
            )
            tag_loss = tag_loss / batch["batch_num_tokens"]
            return span_loss + tag_loss + commit_loss

    def tetra_to_tags(self, batch, cats):
        assert self.back_cycle is not None and self.back_use_gold_trees

        # tree to tags
        encoder_in = F.one_hot(batch["tetra_tags"].to(self.device), num_classes=len(
            self.tetra_tag_system.tag_vocab) + 1).float()
        encoder_in = self.back_project(encoder_in)
        encoder_in = self.back_add_timing(self.morpho_emb_dropout(encoder_in))

        annotations = self.back_cycle(
            encoder_in, batch['tetra_padding_mask'].to(self.device))

        logits = self.f_back(annotations)
        loss = self.back_criterion(
            logits[batch["tetra_tags_mask"].to(self.device)], cats[batch["valid_token_mask"].to(self.device)]) / 2

        # back prop experiment
        loss += self.back_criterion(
            cats[batch["valid_token_mask"].to(self.device)], logits[batch["tetra_tags_mask"].to(self.device)]) / 2

        # is it right for cats to be after the gumbel softmax? use logits v logits

        # for later -- use this as a prior for inference
        if np.random.random() < 0.001:
            print('back loss: ', loss.data.cpu().numpy())

        return self.back_loss_constant * loss

    def tetra_to_annotations(self, batch, eca):
        assert self.back_cycle is not None and self.back_use_gold_trees

        # tree to tags
        encoder_in = F.one_hot(batch["tetra_tags"].to(self.device), num_classes=len(
            self.tetra_tag_system.tag_vocab) + 1).float()
        encoder_in = self.back_project(encoder_in)
        encoder_in = self.back_add_timing(self.morpho_emb_dropout(encoder_in))

        annotations = self.back_cycle(
            encoder_in, batch['tetra_padding_mask'].to(self.device))
        logits = self.f_back(annotations)
        loss = self.back_criterion(
            logits[batch["tetra_tags_mask"].to(self.device)], eca[batch["valid_token_mask"].to(self.device)]) / 2

        # back prop experiment
        loss += self.back_criterion(
            eca[batch["valid_token_mask"].to(self.device)], logits[batch["tetra_tags_mask"].to(self.device)]) / 2

        # for later -- use this as a prior for inference
        if np.random.random() < 0.001:
            print('back loss: ', loss.data.cpu().numpy())

        return self.back_loss_constant * loss

    def predicted_tetra_to_tags(self, batch, cats, span_scores):
        assert False, 'using gold trees for now'
        assert self.back_cycle is not None and not self.back_use_gold_trees

        lengths = batch["valid_token_mask"].sum(-1) - 2
        lengths = lengths.to(span_scores.device)
        charts_np = self.decoder.charts_from_pytorch_scores_batched(
            span_scores, lengths
        )
        tetra_tags = []
        tetra_tags_mask = []
        for i in range(len(lengths)):
            tree = self.decoder.tree_from_chart(
                charts_np[i], leaves=[('S', str(i)) for i in range(lengths[i])])
            tree = self.tree_transform(tree)
            tets = [len(self.tetra_tag_system.tag_vocab)] + self.tetra_tag_system.ids_from_tree(
                tree) + [len(self.tetra_tag_system.tag_vocab)]
            tetra_tags.append(torch.Tensor(tets))
            mask = [i in self.tetra_leaves for i in tets]
            mask[0] = mask[-1] = True
            tetra_tags_mask.append(torch.Tensor(mask).bool())

        tetra_tags = nn.utils.rnn.pad_sequence(
            tetra_tags,
            batch_first=True,
            padding_value=len(self.tetra_tag_system.tag_vocab)
        ).long()

        tetra_tags_mask = nn.utils.rnn.pad_sequence(
            tetra_tags_mask,
            batch_first=True
        )

        # tree to tags
        encoder_in = F.one_hot(tetra_tags.to(self.device), num_classes=len(
            self.tetra_tag_system.tag_vocab) + 1).float()
        encoder_in = self.back_project(encoder_in)
        encoder_in = self.add_timing(self.morpho_emb_dropout(encoder_in))
        annotations = self.back_cycle(encoder_in, None)
        logits = self.f_back(annotations)
        loss = self.back_criterion(
            logits[tetra_tags_mask.to(self.device)], cats[batch["valid_token_mask"].to(self.device)])

        print('back loss: ', loss.data.cpu().numpy())

        return self.back_loss_constant * loss

    def _parse_encoded(
        self, examples, encoded, return_compressed=False, return_scores=False, return_cats=False, tau=0.0
    ):
        with torch.no_grad():
            if self.check_force_cats(examples):
                span_scores, tag_scores, categories, _ = self.forward(
                    batch=None, force_cats=examples)
                lengths = np.array([len(example) - 2 for example in examples])
            else:
                batch = self.pad_encoded(encoded)
                span_scores, tag_scores, categories, _ = self.forward(batch, tau)
                lengths = batch["valid_token_mask"].sum(-1) - 2
                lengths = lengths.to(span_scores.device)

            if return_scores:
                span_scores_np = span_scores.cpu().numpy()
            else:
                # Start/stop tokens don't count, so subtract 2
                charts_np = self.decoder.charts_from_pytorch_scores_batched(
                    span_scores, lengths
                )
            if tag_scores is not None:
                tag_ids_np = tag_scores.argmax(-1).cpu().numpy()
            else:
                tag_ids_np = None

        if self.check_force_cats(examples):
            for i in range(len(examples)):
                yield self.decoder.tree_from_chart(charts_np[i], leaves=[str(cat) for cat in examples[i][1:-1]])
            return

        for i in range(len(encoded)):
            example_len = len(examples[i].words)
            if return_scores:
                yield span_scores_np[i, :example_len, :example_len]
            elif return_compressed:
                output = self.decoder.compressed_output_from_chart(
                    charts_np[i])
                if tag_ids_np is not None:
                    output = output.with_tags(
                        tag_ids_np[i, 1: example_len + 1])
                yield output
            else:
                if tag_scores is None:
                    leaves = examples[i].pos()
                else:
                    predicted_tags = [
                        self.tag_from_index[i]
                        for i in tag_ids_np[i, 1: example_len + 1]
                    ]
                    leaves = [
                        (word, predicted_tag)
                        for predicted_tag, (word, gold_tag) in zip(
                            predicted_tags, examples[i].pos()
                        )
                    ]
                if return_cats:
                    yield self.decoder.tree_from_chart(charts_np[i], leaves=leaves), categories[i]
                else:
                    yield self.decoder.tree_from_chart(charts_np[i], leaves=leaves)

    def check_force_cats(self, examples):
        return self.d_cats > 0 and (examples is None or isinstance(examples[0], (list, np.ndarray)))

    def parse(
        self,
        examples,
        return_compressed=False,
        return_scores=False,
        subbatch_max_tokens=None,
        return_cats=False,
        tau=0.0
    ):
        training = self.training
        self.eval()
        if self.check_force_cats(examples):
            encoded = None
        else:
            encoded = [self.encode(example) for example in examples]
        if subbatch_max_tokens is not None:
            res = subbatching.map(
                self._parse_encoded,
                examples,
                encoded,
                costs=self._get_lens(encoded),
                max_cost=subbatch_max_tokens,
                return_compressed=return_compressed,
                return_scores=return_scores,
                return_cats=return_cats,
                tau=tau
            )
        else:
            res = self._parse_encoded(
                examples,
                encoded,
                return_compressed=return_compressed,
                return_scores=return_scores,
                return_cats=return_cats,
                tau=tau
            )
            # fixing determinism bug
            res = list(res)
        self.train(training)
        return res
