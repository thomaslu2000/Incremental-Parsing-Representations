import argparse
import functools
import itertools
import os.path
import os
import time

import torch

import numpy as np

from benepar2 import labeler
from benepar2 import decode_chart
from benepar2 import nkutil
from benepar2 import parse_chart
import evaluate
import learning_rates
import treebanks
from tree_transforms import collapse_unlabel_binarize, random_parsing_subspan


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def make_hparams():
    return nkutil.HParams(
        # Data processing
        max_len_train=0,  # no length limit
        max_len_dev=0,  # no length limit
        train_size=0,
        dev_size=0,
        # Optimization
        batch_size=16,
        lr=0.001,
        pretrained_lr=0.00005,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0.0,  # no clipping
        checks_per_epoch=4,
        step_decay_factor=0.5,
        step_decay_patience=5,
        max_consecutive_decays=3,  # establishes a termination criterion
        # Partitioned transformer encoder
        encoder_gum=False,
        use_encoder=False,
        d_model=100,
        num_layers=2,
        num_heads=6,
        d_kv=32,
        d_ff=100,
        encoder_max_len=512,
        # Dropout
        morpho_emb_dropout=0.2,
        attention_dropout=0.2,
        relu_dropout=0.1,
        residual_dropout=0.2,
        # Output heads and losses
        force_root_constituent="false",  # "auto",
        predict_tags=False,
        d_label_hidden=256,
        d_tag_hidden=256,
        tag_loss_scale=5.0,
    )


def run_train(args, hparams):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    # Make sure that pytorch is actually being initialized randomly.
    # On my cluster I was getting highly correlated results from multiple
    # runs, but calling reset_parameters() changed that. A brief look at the
    # pytorch source code revealed that pytorch initializes its RNG by
    # calling std::random_device, which according to the C++ spec is allowed
    # to be deterministic.
    seed_from_numpy = np.random.randint(2147483648)
    print("Manual seed for pytorch:", seed_from_numpy)
    torch.manual_seed(seed_from_numpy)

    hparams.set_from_args(args)
    print("Hyperparameters:")
    hparams.print()

    # Using penn treebank as of now for testing

    # print("Constructing vocabularies...")
    # op_trees = train_treebank.trees
    # label_vocab = decode_chart.ChartDecoder.build_vocab(op_trees)
    # char_vocab = None
    # print('Label Vocab Size:', len(label_vocab))

    # tag_vocab = set()
    # for tree in op_trees:
    #     for _, tag in tree.pos():
    #         tag_vocab.add(tag)
    # tag_vocab = ["UNK"] + sorted(tag_vocab)
    # tag_vocab = {label: i for i, label in enumerate(tag_vocab)}

    # del op_trees

    if hparams.force_root_constituent.lower() in ("true", "yes", "1"):
        hparams.force_root_constituent = True
    elif hparams.force_root_constituent.lower() in ("false", "no", "0"):
        hparams.force_root_constituent = False

    print("Initializing model...")
    parser = labeler.Labeler(hparams)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = treebanks.load_trees(
        args.train_path, args.train_path_text, args.text_processing
    )
    if hparams.max_len_train > 0:
        train_treebank = train_treebank.filter_by_length(hparams.max_len_train)
    train_treebank = [parser.process.simple_tags_and_labels(
        tree) for tree in train_treebank.trees[:hparams.train_size or None]]
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = treebanks.load_trees(
        args.dev_path, args.dev_path_text, args.text_processing
    )
    if hparams.max_len_dev > 0:
        dev_treebank = dev_treebank.filter_by_length(hparams.max_len_dev)
    dev_data = [parser.process.simple_tags_and_labels(
        tree, return_tetra=True) for tree in dev_treebank.trees[:hparams.dev_size or None]]
    print('Dev_data size:', len(dev_data))
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    if args.parallelize:
        parser.parallelize()
    elif torch.cuda.is_available():
        parser.cuda()
    else:
        print("Not using CUDA!")

    print("Initializing optimizer...")

    trainable_parameters = [
        p for p in parser.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=hparams.lr, betas=(0.9, 0.98), eps=1e-9
    )

    scheduler = learning_rates.WarmupThenReduceLROnPlateau(
        optimizer,
        hparams.learning_rate_warmup_steps,
        mode="max",
        factor=hparams.step_decay_factor,
        patience=hparams.step_decay_patience * hparams.checks_per_epoch,
        verbose=True,
    )

    clippable_parameters = trainable_parameters
    grad_clip_threshold = (
        np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm
    )

    print("Training...")
    total_processed = 0
    current_processed = 0
    check_every = len(train_treebank) / hparams.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None
    best_dev_processed = 0

    start_time = time.time()

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path
        nonlocal best_dev_processed

        dev_start_time = time.time()

        dev_predicted = parser.make_trees(dev_data)

        dev_fscore = evaluate.evalb(
            args.evalb_dir, dev_treebank.trees[:hparams.dev_size or None], dev_predicted)

        print(
            "dev-fscore {} "
            "best-dev {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                best_dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore
            )
            best_dev_processed = total_processed
            print("Saving new best model to {}...".format(best_dev_model_path))
            torch.save(
                {
                    "config": parser.config,
                    "state_dict": parser.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                best_dev_model_path + ".pt",
            )
        return None

    data_loader = torch.utils.data.DataLoader(
        train_treebank,
        batch_size=hparams.batch_size,
        shuffle=True,
        collate_fn=functools.partial(
            parser.encode_and_collate_subbatches,
            subbatch_max_tokens=args.subbatch_max_tokens,
        ),
    )

    iteration = 0

    # dist = check_dev()

    for epoch in itertools.count(start=1):
        epoch_start_time = time.time()

        for batch_num, batch in enumerate(data_loader, start=1):
            iteration += 1
            optimizer.zero_grad()
            parser.train()

            batch_loss_value = 0.0
            accs = np.array([])
            for subbatch_size, subbatch in batch:
                loss, acc = parser.compute_loss(subbatch)
                accs = np.concatenate([accs, acc])
                loss_value = float(loss.data.cpu().numpy())
                batch_loss_value += loss_value
                if loss_value > 0:
                    loss.backward()
                del loss
                total_processed += subbatch_size
                current_processed += subbatch_size

            grad_norm = torch.nn.utils.clip_grad_norm_(
                clippable_parameters, grad_clip_threshold
            )

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "train acc {:.3f} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "grad-norm {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    batch_num,
                    int(np.ceil(len(train_treebank) / hparams.batch_size)),
                    np.mean(accs),
                    total_processed,
                    batch_loss_value,
                    grad_norm,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            optimizer.step()
            if current_processed >= check_every:

                current_processed -= check_every
                check_dev()
                scheduler.step(metrics=best_dev_fscore)
            else:
                scheduler.step()

        if (total_processed - best_dev_processed) > (
            (hparams.step_decay_patience + 1)
            * hparams.max_consecutive_decays
            * len(train_treebank)
        ):
            print("Terminating due to lack of improvement in dev fscore.")
            break


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument(
        "--train-path", default="data/02-21.10way.clean")
    subparser.add_argument("--train-path-text", type=str)
    subparser.add_argument("--dev-path", default="data/22.auto.clean")
    subparser.add_argument("--dev-path-text", type=str)
    subparser.add_argument("--text-processing", default="default")
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--parallelize", action="store_true")
    subparser.add_argument("--print-vocabs", action="store_true")

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
