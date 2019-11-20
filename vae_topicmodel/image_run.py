# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import yaml
import argparse

import random
import numpy as np
import tensorflow as tf

from vae_topicmodel.models import Model
from vae_topicmodel.image_reader import ImageReader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-file", default=None, help="The config file used.")
    parser.add_argument("--model", default=None, choices=Model.available_models,
                        help="The model used, override what is defined in cfg file.")
    parser.add_argument("--topic-dim", default=50, type=int,
                        help="The topic dimension, override what is defined in cfg file.")
    parser.add_argument("--train-cfg-file", default=None, help="The training config file used.")
    parser.add_argument("--dataset", metavar="DATASET", default="20news",
                        help="The name of the dataset used,"
                        "the dataset should be put under directory `<project root>/datasets/<DATASET>`.")
    parser.add_argument("--save", default=None, metavar="SAVE_PATH",
                        help="Save the model to path. If not specified, the model will not be saved.")
    parser.add_argument("--snapshot-dir", default=None,
                        help="Snapshot model to path. If not specified, default to `os.path.join(os.path.dirname(SAVE_PATH), 'snapshots')`")
    parser.add_argument("--save-tensor-dir", default=None,
                        help="Save tensor to path. If not specified, default to `os.path.join(os.path.dirname(SAVE_PATH), 'tensors')`")
    parser.add_argument("--summary-dir", default=None,
                        help="The optional summary dir for VAE models to write summary to.")
    parser.add_argument("--seed", default=None, type=int,
                        help="The random seed used for random libs.")
    parser.add_argument("--keep-labels", default=None,
                        help="Comma-split list of label indexes that will be kept(eg. --keep-labels 0,1,2,3). By default, all labels will be kept.")
    parser.add_argument("--add-valid-to-train", default=False, action="store_true",
                        help="Do not use hold out valid, add valid data to train.")
    parser.add_argument("--floatX", type=int, default=32)

    args = parser.parse_args()
    if not args.save:
        print("WARN: The model will not be saved if `--save` argument is not given.")
    else:
        print("The model will be saved to ", args.save)
        if not args.snapshot_dir:
            args.snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(args.save)), "snapshots")
        if not args.save_tensor_dir:
            args.save_tensor_dir = os.path.join(os.path.dirname(os.path.abspath(args.save)), "tensors")
    print("Snapshot will be saved to ", args.snapshot_dir)

    tf.floatX = getattr(tf, "float{}".format(args.floatX))
    np.floatX = getattr(np, "float{}".format(args.floatX))

    if args.summary_dir:
        print("The training summaries will be saved to ", args.summary_dir)

    if args.seed:
        print("Set random seed to {}".format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    # Construct dataset reader
    reader = ImageReader(args.dataset, add_valid_to_train=args.add_valid_to_train)
    if args.keep_labels is not None:
        reader.filter_data_labels([int(l) for l in args.keep_labels.split(",")])

    if args.cfg_file is None:
        assert args.model and args.topic_dim, "Require `--model` and `--topic-dim` argument if `--cfg-file` not specified."
        cfg = {"model": args.model,
               "topic_dim": args.topic_dim}
    else:
        cfg = yaml.load(open(args.cfg_file, "r").read())
        if args.model:
            cfg["model"] = args.model
        if args.topic_dim:
            cfg["topic_dim"] = args.topic_dim
        assert "topic_dim" in cfg and "model" in cfg, "`model` and `topic_dim` must be specified via config file or `--topic-dim`."

    print("topic dim: {}; vocab dim: {}".format(cfg["topic_dim"], reader.vocab_dim))
    train_cfg = yaml.load(open(args.train_cfg_file, "r").read()) if args.train_cfg_file else {}

    cfg["reader"] = reader
    cfg["vocab_dim"] = reader.vocab_dim
    cfg["summary_dir"] = args.summary_dir
    train_cfg["snapshot_dir"] = args.snapshot_dir
    train_cfg["save_tensor_dir"] = args.save_tensor_dir

    model = Model.init_model_from_cfg(cfg, train_cfg)
    model.init()
    model.train()

    # Test the loss on test set
    loss = model.test_on_dataset()
    print("Test loss: ", loss)

    if args.save:
        model.save(args.save)

if __name__ == "__main__":
    main()
