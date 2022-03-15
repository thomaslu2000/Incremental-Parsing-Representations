# Experiments

This file contains commands that can be used to replicate the best parsing numbers that we reported in our papers. See `data/README.md` for information on how to prepare the required datasets.

Without pre-training:

```bash
python src/main.py train \
    --train-path "data/wsj/train_02-21.LDC99T42" \
    --dev-path "data/wsj/dev_22.LDC99T42" \
    --use-chars-lstm --use-encoder --num-layers 8 \
    --batch-size 250 --learning-rate 0.0008 \
    --model-path-base models/English_charlstm
```

With BERT (single model, large, uncased):

```bash
python src/main.py train \
    --train-path "data/wsj/train_02-21.LDC99T42" \
    --dev-path "data/wsj/dev_22.LDC99T42" \
    --use-pretrained --pretrained-model "bert-large-uncased" \
    --predict-tags \
    --model-path-base models/English_bert_large_uncased
```

To evaluate:

```bash
python src/main.py test \
    --test-path "data/wsj/test_23.LDC99T42" \
    --no-predict-tags \
    --model-path models/English_bert_large_uncased_*.pt
```
