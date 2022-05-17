# Experiments

This file contains commands that can be used to replicate the best parsing numbers that we reported in our papers. See `data/README.md` for information on how to prepare the required datasets.


```bash
python -u src/main.py train \
    --train-path data/02-21.10way.clean \
    --dev-path data/22.auto.clean \
    --use-pretrained --pretrained-model "gpt2-medium" \
    --use-encoder \
    --pretrained-divide 20.0 \
    --discrete-cats 512 \
    --use-vq \
    --num-layers 8 \
    --bpe-dropout 0.2 \
    --batch-size 32 \
    --learning-rate 3e-5 \
    --novel-learning-rate 1e-4 \
    --vq-wait-steps 1245 \
    --vq-observe-steps 1245 \
    --vq-interpolate-steps 1245 \
    --model-path-base /models/512.pt \
    --evalb-dir EVALB_LABELED/
```

With BERT (single model, large, uncased):

```bash
python -u src/main.py train \
    --train-path data/02-21.10way.clean \
    --dev-path data/22.auto.clean \
    --use-pretrained --pretrained-model "bert-large-uncased" \
    --subbatch-max-tokens 500 \
    --use-encoder \
    --bpe-dropout 0.2 \
    --pretrained-divide 1.0 \
    --discrete-cats 512 \
    --use-vq \
    --num-layers 8 \
    --batch-size 32 \
    --learning-rate 3e-5 \
    --novel-learning-rate 1e-4 \
    --vq-wait-steps 1245 \
    --vq-observe-steps 1245 \
    --vq-interpolate-steps 1245 \
    --model-path-base /models/bert_512.pt \
    --evalb-dir EVALB/
```

To evaluate:

```bash
python src/main.py test \
    --model-path /models/512.pt
    --test-path data/23.auto.clean \
    --subbatch-max-tokens 750 \
    --evalb-dir EVALB/
```
