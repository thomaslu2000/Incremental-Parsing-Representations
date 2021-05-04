import argparse
import functools
import itertools
import os.path
import os
import time

import torch
import torch.nn.functional as F

import numpy as np

from benepar import char_lstm
from benepar import decode_chart
from benepar import nkutil
from benepar import parse_chart
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
        # Cycle consistency
        back_cycle=False,
        back_layers=4,
        back_loss_constant=1.0,
        back_use_gold_trees=True,
        back_loss_type='kl',
        # Discrete gumbel
        use_vq=False,
        vq_decay=0.97,
        vq_commitment=0.1,
        vq_coreset_size_multiplier=10,
        vq_wait_steps=1245,
        vq_observe_steps=1245,
        vq_interpolate_steps=1245,
        discrete_cats=0,
        tau=3.0,
        anneal_rate=2e-5,
        en_tau=3.0,
        en_anneal_rate=2e-5,
        tau_min=0.05,
        pretrained_divide=1.0,
        encoder_gum=False,
        tags_per_word=1,
        tag_combine_start=np.inf,
        tag_combine_interval=300,
        tag_combine_mask_thres=0.05,
        tag_split_thres=1.001,  # disabled by default
        # Data processing
        two_label_subspan=False,
        two_label=False,
        max_len_train=0,  # no length limit
        max_len_dev=0,  # no length limit
        # Optimization
        batch_size=32,
        novel_learning_rate=0.,  # don't use separate learning rate
        learning_rate=0.00005,
        pretrained_lr=0.00005,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0.0,  # no clipping
        checks_per_epoch=4,
        step_decay_factor=0.5,
        step_decay_patience=5,
        max_consecutive_decays=3,  # establishes a termination criterion
        # CharLSTM
        use_chars_lstm=False,
        d_char_emb=64,
        char_lstm_input_dropout=0.2,
        # BERT and other pre-trained models
        use_pretrained=False,
        pretrained_model="bert-base-uncased",
        use_forced_lm=False,
        # Partitioned transformer encoder
        tag_dist='',
        uni=False,
        all_layers_uni=False,
        first_heads=-1,
        use_encoder=False,
        d_model=1024,
        num_layers=8,
        num_heads=8,
        d_kv=64,
        d_ff=2048,
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
    print("Loading training trees from {}...".format(args.train_path))
    print(args.train_path, args.train_path_text, args.text_processing)
    train_treebank = treebanks.load_trees(
        args.train_path, args.train_path_text, args.text_processing
    )
    if hparams.max_len_train > 0:
        train_treebank = train_treebank.filter_by_length(hparams.max_len_train)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = treebanks.load_trees(
        args.dev_path, args.dev_path_text, args.text_processing
    )
    if hparams.max_len_dev > 0:
        dev_treebank = dev_treebank.filter_by_length(hparams.max_len_dev)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    hparams.tree_transform = None

    if hparams.two_label:
        hparams.tree_transform = collapse_unlabel_binarize

    if hparams.tree_transform is not None:
        for treebank in [train_treebank, dev_treebank]:
            for parsing_example in treebank.examples:
                parsing_example.tree = hparams.tree_transform(
                    parsing_example.tree)

    print("Constructing vocabularies...")
    op_trees = [tree if hparams.tree_transform is None else hparams.tree_transform(
        tree) for tree in train_treebank.trees]
    label_vocab = decode_chart.ChartDecoder.build_vocab(op_trees)
    if hparams.use_chars_lstm:
        char_vocab = char_lstm.RetokenizerForCharLSTM.build_vocab(
            train_treebank.sents)
    else:
        char_vocab = None
    print('Label Vocab Size:', len(label_vocab))

    tag_vocab = set()
    for tree in op_trees:
        for _, tag in tree.pos():
            tag_vocab.add(tag)
    tag_vocab = ["UNK"] + sorted(tag_vocab)
    tag_vocab = {label: i for i, label in enumerate(tag_vocab)}

    del op_trees

    if hparams.two_label_subspan:
        for parsing_example in dev_treebank:
            parsing_example.tree = collapse_unlabel_binarize(
                parsing_example.tree)
        hparams.tree_transform = collapse_unlabel_binarize
        hparams.two_label_subspan = random_parsing_subspan

    if hparams.force_root_constituent.lower() in ("true", "yes", "1"):
        hparams.force_root_constituent = True
    elif hparams.force_root_constituent.lower() in ("false", "no", "0"):
        hparams.force_root_constituent = False
    elif hparams.force_root_constituent.lower() == "auto":
        hparams.force_root_constituent = (
            decode_chart.ChartDecoder.infer_force_root_constituent(
                train_treebank.trees)
        )
        print("Set hparams.force_root_constituent to",
              hparams.force_root_constituent)

    print("Initializing model...")
    parser = parse_chart.ChartParser(
        tag_vocab=tag_vocab,
        label_vocab=label_vocab,
        char_vocab=char_vocab,
        hparams=hparams,
    )
    if args.parallelize:
        parser.parallelize()
    elif torch.cuda.is_available():
        parser.cuda()
    else:
        print("Not using CUDA!")

    print("Initializing optimizer...")

    pretrained_weights = list(
        params for params in parser.pretrained_model.parameters() if params.requires_grad)
    other_weights = []
    for p in parser.parameters():
        if p.requires_grad and all(p is not p2 for p2 in pretrained_weights):
            other_weights.append(p)

    trainable_parameters = [
        {'params': pretrained_weights, 'lr': hparams.pretrained_lr},
        {'params': other_weights}
    ]
    # trainable_parameters = list(
    #     params for params in parser.parameters() if params.requires_grad)

    if hparams.novel_learning_rate == 0.0:
        optimizer = torch.optim.Adam(
            trainable_parameters, lr=hparams.learning_rate, betas=(0.9, 0.98), eps=1e-9
        )
        base_lr = hparams.learning_rate
    else:
        trainable_parameters = list(
            params for params in parser.parameters() if params.requires_grad)

        pretrained_params = set(trainable_parameters) & set(
            parser.pretrained_model.parameters())
        novel_params = set(trainable_parameters) - pretrained_params
        grouped_trainable_parameters = [
            {
                'params': list(pretrained_params),
                'lr': hparams.learning_rate,
            },
            {
                'params': list(novel_params),
                'lr': hparams.novel_learning_rate,
            },
        ]
        optimizer = torch.optim.Adam(
            grouped_trainable_parameters, lr=hparams.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        base_lr = min(hparams.learning_rate, hparams.novel_learning_rate)

    scheduler = learning_rates.WarmupThenReduceLROnPlateau(
        optimizer,
        hparams.learning_rate_warmup_steps,
        mode="max",
        factor=hparams.step_decay_factor,
        patience=hparams.step_decay_patience * hparams.checks_per_epoch,
        verbose=True,
    )

    clippable_parameters = pretrained_weights + other_weights
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

        dev_predicted_and_cats = parser.parse(
            dev_treebank.without_gold_annotations(),
            subbatch_max_tokens=args.subbatch_max_tokens,
            tau=0.0,
            en_tau=0.0,
            return_cats=hparams.discrete_cats != 0,
            return_encoded=hparams.tag_dist
        )
        if hparams.tag_dist:
            dev_predicted_and_cats, encoded = dev_predicted_and_cats

        if hparams.discrete_cats == 0:
            dev_predicted = dev_predicted_and_cats
            dist = None
        else:
            dist = torch.zeros(hparams.discrete_cats)
            dev_predicted = []
            for dev_tree, cat in dev_predicted_and_cats:
                dev_predicted.append(dev_tree)
                if len(cat.shape) == 1:
                    cat = F.one_hot(torch.tensor(cat), hparams.discrete_cats)
                dist += cat.sum(dim=0).cpu()
            dist /= dist.sum()

        print(dist)

        dev_fscore = evaluate.evalb(
            args.evalb_dir, dev_treebank.trees, dev_predicted)

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

            # if hparams.tag_dist:
            #     with open(hparams.tag_dist, 'wb') as f:
            #         np.save(f, tag_distribution)

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
        if dist is None:
            return dist
        return dist.cpu().numpy()

    data_loader = torch.utils.data.DataLoader(
        train_treebank,
        batch_size=hparams.batch_size,
        shuffle=True,
        collate_fn=functools.partial(
            parser.encode_and_collate_subbatches,
            subbatch_max_tokens=args.subbatch_max_tokens,
        ),
    )

    tau = hparams.tau
    en_tau = hparams.tau
    tag_combine_start = hparams.tag_combine_start
    iteration = 0

    # dist = check_dev()

    for epoch in itertools.count(start=1):
        epoch_start_time = time.time()

        for batch_num, batch in enumerate(data_loader, start=1):
            iteration += 1
            optimizer.zero_grad()
            parser.commit_loss_accum = 0.0
            parser.train()

            if hparams.use_vq and hparams.vq_interpolate_steps == 0:
                tau = 0.0
            elif hparams.use_vq:
                step = (total_processed // hparams.batch_size) - (
                    hparams.vq_wait_steps + hparams.vq_observe_steps)
                if step < 0:
                    tau = 1.0
                elif step >= hparams.vq_interpolate_steps:
                    tau = 0.0
                else:
                    tau = max(0.0, 1.0 - step / hparams.vq_interpolate_steps)

            steps_past_warmup = (total_processed // hparams.batch_size
                                 ) - hparams.learning_rate_warmup_steps
            if steps_past_warmup > 0:
                current_lr = min([g["lr"] for g in optimizer.param_groups])
                new_vq_decay = 1.0 - (
                    (1.0 - hparams.vq_decay) * (current_lr / base_lr))
                if hparams.use_vq and new_vq_decay != parser.vq.decay:
                    parser.vq.decay = new_vq_decay
                    print("Adjusted vq decay to:", new_vq_decay)

            batch_loss_value = 0.0
            for subbatch_size, subbatch in batch:
                loss = parser.compute_loss(subbatch, tau=tau, en_tau=en_tau)
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

            optimizer.step()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "grad-norm {:.4f} "
                "commit-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    batch_num,
                    int(np.ceil(len(train_treebank) / hparams.batch_size)),
                    total_processed,
                    batch_loss_value,
                    grad_norm,
                    float(parser.commit_loss_accum),
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                dist = check_dev()
                scheduler.step(metrics=best_dev_fscore)

                if (hparams.discrete_cats > 0 and hparams.use_vq):
                    ptdist = parser.vq.cluster_size.cpu().numpy()
                    if np.sum(ptdist) > 0:
                        ptdist = ptdist / np.sum(ptdist)
                    num_categories_in_use = np.sum(ptdist > 1e-20)
                    print("Number of categories in use:", num_categories_in_use)

                if hparams.discrete_cats > 0 and not hparams.use_vq:
                    # Gumbel temperature annealing
                    tau = np.maximum(
                        hparams.tau * np.exp(-hparams.anneal_rate * iteration), hparams.tau_min)
                    en_tau = np.maximum(
                        hparams.en_tau * np.exp(-hparams.en_anneal_rate * iteration), hparams.tau_min)
                    print('setting temperature to: {:.4f}, hard attention tau to {:.3f}'.format(
                        tau, en_tau))
            else:
                scheduler.step()

        if (total_processed - best_dev_processed) > (
            (hparams.step_decay_patience + 1)
            * hparams.max_consecutive_decays
            * len(train_treebank)
        ):
            print("Terminating due to lack of improvement in dev fscore.")
            break


def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = treebanks.load_trees(
        args.test_path, args.test_path_text, args.text_processing
    )
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    if len(args.model_path) != 1:
        raise NotImplementedError(
            "Ensembling multiple parsers is not "
            "implemented in this version of the code."
        )

    model_path = args.model_path[0]
    print("Loading model from {}...".format(model_path))
    parser = parse_chart.ChartParser.from_trained(model_path)
    if args.parallelize:
        parser.parallelize()
    elif torch.cuda.is_available():
        parser.cuda()

    print("Parsing test sentences...")
    start_time = time.time()

    test_predicted = parser.parse(
        test_treebank.without_gold_annotations(),
        subbatch_max_tokens=args.subbatch_max_tokens,
    )

    if args.output_path == "-":
        for tree in test_predicted:
            print(tree.pformat(margin=1e100))
    elif args.output_path:
        with open(args.output_path, "w") as outfile:
            for tree in test_predicted:
                outfile.write("{}\n".format(tree.pformat(margin=1e100)))

    # The tree loader does some preprocessing to the trees (e.g. stripping TOP
    # symbols or SPMRL morphological features). We compare with the input file
    # directly to be extra careful about not corrupting the evaluation. We also
    # allow specifying a separate "raw" file for the gold trees: the inputs to
    # our parser have traces removed and may have predicted tags substituted,
    # and we may wish to compare against the raw gold trees to make sure we
    # haven't made a mistake. As far as we can tell all of these variations give
    # equivalent results.
    ref_gold_path = args.test_path
    if args.test_path_raw is not None:
        print("Comparing with raw trees from", args.test_path_raw)
        ref_gold_path = args.test_path_raw

    test_fscore = evaluate.evalb(
        args.evalb_dir, test_treebank.trees, test_predicted, ref_gold_path=ref_gold_path
    )

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )


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
        "--train-path", default="data/wsj/train_02-21.LDC99T42")
    subparser.add_argument("--train-path-text", type=str)
    subparser.add_argument("--dev-path", default="data/wsj/dev_22.LDC99T42")
    subparser.add_argument("--dev-path-text", type=str)
    subparser.add_argument("--text-processing", default="default")
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--parallelize", action="store_true")
    subparser.add_argument("--print-vocabs", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path", nargs="+", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", default="data/wsj/test_23.LDC99T42")
    subparser.add_argument("--test-path-text", type=str)
    subparser.add_argument("--test-path-raw", type=str)
    subparser.add_argument("--text-processing", default="default")
    subparser.add_argument("--subbatch-max-tokens", type=int, default=500)
    subparser.add_argument("--parallelize", action="store_true")
    subparser.add_argument("--output-path", default="")

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
