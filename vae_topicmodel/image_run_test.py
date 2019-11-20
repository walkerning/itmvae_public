# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import yaml
import argparse

import tensorflow as tf
import numpy as np

from vae_topicmodel.models import Model
from vae_topicmodel.image_reader import ImageReader
from vae_topicmodel.test import load_test_modules

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("load", help="Load the model from path.")
    parser.add_argument("--cfg-file", default=None, help="The config file used.")
    parser.add_argument("--train-cfg-file", default=None, help="The training config file, only test configurations will be used.")
    parser.add_argument("--topic-dim", default=50, type=int,
                        help="The topic dimension, override what is defined in cfg file.")
    parser.add_argument("--dataset", metavar="DATASET", default="mnist",
                        help="The name of the dataset used,"
                        "the dataset should be put under directory `<project root>/datasets/<DATASET>`.")
    parser.add_argument("--keep-labels", default=None,
                        help="Comma-split list of label indexes that will be kept(eg. --keep-labels 0,1,2,3). By default, all labels will be kept.")
    parser.add_argument("--mc-samples", default=100, type=int)
    parser.add_argument("--floatX", type=int, default=32)

    available_test_modules = {}
    for modname, mod in load_test_modules():
        mod.populate_argparser(parser, scope=modname)
        available_test_modules[modname] = mod

    parser.add_argument("-t", "--test-module", action="append",
                        choices=available_test_modules.keys(), default=[])

    args = parser.parse_args()
    args.load = os.path.abspath(args.load)
    print("The model will be loaded from ", args.load)

    tf.floatX = getattr(tf, "float{}".format(args.floatX))
    np.floatX = getattr(np, "float{}".format(args.floatX))

    # Construct dataset reader, test only
    reader = ImageReader(args.dataset)
    if args.keep_labels is not None:
        reader.filter_data_labels([int(l) for l in args.keep_labels.split(",")])

    if args.cfg_file is None:
        # Default
        args.cfg_file = os.path.join(os.path.dirname(args.load), "config/model.yaml")
    if args.train_cfg_file is None:
        # Default
        args.train_cfg_file = os.path.join(os.path.dirname(args.load), "config/train.yaml")
    cfg = yaml.load(open(args.cfg_file, "r").read())
    if args.topic_dim:
        cfg["topic_dim"] = args.topic_dim

    print("topic dim: {}; vocab dim: {}".format(cfg["topic_dim"], reader.vocab_dim))
    train_cfg = yaml.load(open(args.train_cfg_file, "r").read()) if args.train_cfg_file else {}
    cfg["MC_samples"] = args.mc_samples
    cfg["reader"] = reader
    cfg["vocab_dim"] = reader.vocab_dim

    model = Model.init_model_from_cfg(cfg, train_cfg)
    model.init(test_only=True)
    model.load(args.load)

    # Test the loss and marginal likelihood on test set
    loss, log_likelihood = model.test_on_dataset(tensor=[model._test_tensor, model.log_likelihood_tensor])
    print("Test loss: ", loss)
    print("Marginal log likelihood: ", log_likelihood)

    for test_module_name in args.test_module:
        test_module = available_test_modules[test_module_name]
        print("Run test module: {}".format(test_module_name))
        test_module.do_test(model, reader, args)

if __name__ == "__main__":
    main()
