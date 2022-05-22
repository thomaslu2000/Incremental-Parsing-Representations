# Incremental Parsing Representations

A high-accuracy incremental parser implemented in Python. Based on [Learned Incremental Representations for Parsing](https://aclanthology.org/2022.acl-long.220.pdf) from ACL 2022, which is built upon the works of [Constituency Parsing with a Self-Attentive Encoder](https://arxiv.org/abs/1805.01052) from ACL 2018, and with additional changes described in [Multilingual Constituency Parsing with Self-Attention and Pre-Training](https://arxiv.org/abs/1812.11760).

## Notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X99Vbz9pWv1-w-wZoCHzknYl_EU_GXZc?usp=sharing) This notebook gives an example of inference using the model we trained for our paper.


## Contents
1. [Training](#training)
2. [Reproducing Experiments](#reproducing-experiments)
3. [Citation](#citation)
4. [Credits](#credits)

If you are primarily interested in training your own parsing models, skip to the [Training](#training) section of this README.

## Training

Training requires cloning this repository from GitHub. While the model code in `src/benepar` is based on the `benepar` package on PyPI, much of the related code and training and evaluation scripts directly under `src/` are not.

#### Software Requirements for Training
* Python 3.7 or higher.
* [PyTorch](http://pytorch.org/) 1.6.0, or any compatible version.
* All dependencies required by the `benepar` package, including: [NLTK](https://www.nltk.org/) 3.2, [torch-struct](https://github.com/harvardnlp/pytorch-struct) 0.4, [transformers](https://github.com/huggingface/transformers) 4.3.0, or compatible.
* [pytokenizations](https://github.com/tamuhey/tokenizations/) 0.7.2 or compatible.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. If training on the SPMRL datasets, you will need to run `make` inside the `EVALB_SPMRL/` directory instead.
* [clusopt](https://github.com/giuliano-oliveira/clusopt)
* [torch_struct](http://nlp.seas.harvard.edu/pytorch-struct/)

### Training Instructions

A new model can be trained using the command `python src/main.py train ...`. Some of the available arguments are:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base to use for saving models | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--train-path` | Path to training trees | `data/wsj/train_02-21.LDC99T42`
`--train-path-text` | Optional non-destructive tokenization of the training data | Guess raw text; see `--text-processing`
`--dev-path` | Path to development trees | `data/wsj/dev_22.LDC99T42`
`--dev-path-text` | Optional non-destructive tokenization of the development data | Guess raw text; see `--text-processing`
`--text-processing` | Heuristics for guessing raw text from descructively tokenized tree files. See `load_trees()` in `src/treebanks.py` | Default rules for languages other than Arabic, Chinese, and Hebrew
`--subbatch-max-tokens` | Maximum number of tokens to process in parallel while training (a full batch may not fit in GPU memory) | 2000
`--parallelize` | Distribute pre-trained model (e.g. T5) layers across multiple GPUs. | Use at most one GPU
`--batch-size` | Number of examples per training update | 32
`--checks-per-epoch` | Number of development evaluations per epoch | 4
`--numpy-seed` | NumPy random seed | Random
`--use-pretrained` | Use pre-trained encoder | Do not use pre-trained encoder
`--pretrained-model` | Model to use if `--use-pretrained` is passed. May be a path or a model id from the [HuggingFace Model Hub](https://huggingface.co/models)| `bert-base-uncased`
`--predict-tags` | Adds a part-of-speech tagging component and auxiliary loss to the parser | Do not predict tags
`--use-chars-lstm` | Use learned CharLSTM word representations | Do not use CharLSTM
`--use-encoder` | Use learned transformer layers on top of pre-trained model or CharLSTM | Do not use extra transformer layers
`--num-layers` | Number of transformer layers to use if `--use-encoder` is passed | 8
`--encoder-max-len` | Maximum sentence length (in words) allowed for extra transformer layers | 512
`--use-vq` | Use vector quantization to compress word representations into a smaller set of discrete categories | Do not use vector quantization
`--discrete-cats` | The number of distinct categories used by the read-out network to produce a tree | 0 (Do not use categories)

Additional arguments are available for other hyperparameters; see `make_hparams()` in `src/main.py`. These can be specified on the command line, such as `--num-layers 2` (for numerical parameters), `--predict-tags` (for boolean parameters that default to False), or `--no-XXX` (for boolean parameters that default to True).

For each development evaluation, the F-score on the development set is computed and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. The new filename will be derived from the provided model path base and the development F-score.

Prior to training the parser, you will first need to obtain appropriate training data. We provide [instructions on how to process standard datasets like PTB, CTB, and the SMPRL 2013/2014 Shared Task data](data/README.md). After following the instructions for the English WSJ data, you can use the following command to train an English parser using the default hyperparameters:

```
python src/main.py train --use-pretrained --model-path-base models/en_bert_base
```

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for more examples of good hyperparameter choices.

### Evaluation Instructions

A saved model can be evaluated on a test corpus using the command `python src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path` | Path of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--test-path` | Path to test trees | `data/23.auto.clean`
`--test-path-text` | Optional non-destructive tokenization of the test data | Guess raw text; see `--text-processing`
`--text-processing` | Heuristics for guessing raw text from descructively tokenized tree files. See `load_trees()` in `src/treebanks.py` | Default rules for languages other than Arabic, Chinese, and Hebrew
`--test-path-raw` | Alternative path to test trees that is used for evalb only (used to double-check that evaluation against pre-processed trees does not contain any bugs) | Compare to trees from `--test-path`
`--subbatch-max-tokens` | Maximum number of tokens to process in parallel (a GPU does not have enough memory to process the full dataset in one batch) | 500
`--parallelize` | Distribute pre-trained model (e.g. T5) layers across multiple GPUs. | Use at most one GPU
`--output-path` | Path to write predicted trees to (use `"-"` for stdout). | Do not save predicted trees
`--no-predict-tags` | Use gold part-of-speech tags when running EVALB. This is the standard for publications, and omitting this flag may give erroneously high F1 scores. | Use predicted part-of-speech tags for EVALB, if available

As an example, you can evaluate a trained model using the following command:
```
python src/main.py test --model-path models/en_bert_base_dev=*.pt
```

## Reproducing Experiments

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for instructions on how to reproduce experiments reported in our ACL 2022 paper.

## Citation
Our paper can be cited as follows:

```
@inproceedings{kitaev-etal-2022-learned,
    title = "Learned Incremental Representations for Parsing",
    author = "Kitaev, Nikita  and
      Lu, Thomas  and
      Klein, Dan",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.220",
    pages = "3086--3095",
    abstract = "We present an incremental syntactic representation that consists of assigning a single discrete label to each word in a sentence, where the label is predicted using strictly incremental processing of a prefix of the sentence, and the sequence of labels for a sentence fully determines a parse tree. Our goal is to induce a syntactic representation that commits to syntactic choices only as they are incrementally revealed by the input, in contrast with standard representations that must make output choices such as attachments speculatively and later throw out conflicting analyses. Our learned representations achieve 93.72 F1 on the Penn Treebank with as few as 5 bits per word, and at 8 bits per word they achieve 94.97 F1, which is comparable with other state of the art parsing models when using the same pre-trained embeddings. We also provide an analysis of the representations learned by our system, investigating properties such as the interpretable syntactic features captured by the system and mechanisms for deferred resolution of syntactic ambiguities.",
}
```

## Credits

The code in this repository and portions of this README are based on https://github.com/nikitakit/self-attentive-parser
