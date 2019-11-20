# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import yaml
import argparse

import numpy as np
import tensorflow as tf

from vae_topicmodel.models import Model
from vae_topicmodel.reader import Reader
from vae_topicmodel.test import load_test_modules

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("load", help="Load the model from path.")
    parser.add_argument("--cfg-file", default=None, help="The config file used.")
    parser.add_argument("--train-cfg-file", default=None, help="The training config file, only test configurations will be used.")
    parser.add_argument("--model", default=None, choices=Model.available_models,
                        help="The model used, override what is defined in cfg file.")
    parser.add_argument("--topic-dim", default=None, type=int,
                        help="The topic dimension, override what is defined in cfg file.")
    parser.add_argument("--dataset", metavar="DATASET", default="20news",
                        help="The name of the dataset used,"
                        "the dataset should be put under directory `<project root>/datasets/<DATASET>`.")
    parser.add_argument("--print-topic-file", default=None,
                        help="Optional filename to which the representive words of the topics will be printed to."
                        "If not specified, the topic will be printed to stdout.")
    parser.add_argument("--print-all", default=False, action="store_true", help="print all topics")
    parser.add_argument("--reader-type", default="index", choices=Reader.registry.keys())
    parser.add_argument("--keep-labels", default=None,
                        help="Comma-split list of label indexes that will be kept(eg. --keep-labels 0,1,2,3). By default, all labels will be kept.")
    parser.add_argument("--mc-samples", default=20, type=int)
    parser.add_argument("--no-elbo-test", action="store_true")
    parser.add_argument("--add-valid-to-train", default=False, action="store_true",
                        help="Do not use hold out valid, add valid data to train.")

    available_test_modules = {}
    for modname, mod in load_test_modules():
        mod.populate_argparser(parser, scope=modname)
        available_test_modules[modname] = mod

    parser.add_argument("-t", "--test-module", action="append",
                        choices=available_test_modules.keys(), default=[])

    args = parser.parse_args()
    args.load = os.path.abspath(args.load)
    print("The model will be loaded from ", args.load)

    # Construct dataset reader, test only
    reader = Reader.get_reader_cls(args.reader_type)(args.dataset, add_valid_to_train=args.add_valid_to_train)#, #load=("test",))
    if args.keep_labels is not None:
        reader.filter_data_labels([int(l) for l in args.keep_labels.split(",")])

    if args.cfg_file is None:
        cfg_dir = os.path.join(os.path.dirname(args.load), "config")
        if not os.path.isdir(cfg_dir): # try upper level, assume in `snapshots/` subdirectory
            cfg_dir = os.path.join(os.path.dirname(os.path.dirname(args.load)), "config")
        args.cfg_file = os.path.join(cfg_dir, "model.yaml")
        args.train_cfg_file = os.path.join(cfg_dir, "train.yaml")
    if args.cfg_file is None:
        # NOT USED
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
    cfg["MC_samples"] = args.mc_samples
    cfg["reader"] = reader
    cfg["vocab_dim"] = reader.vocab_dim

    model = Model.init_model_from_cfg(cfg, train_cfg)
    model.init(test_only=True)
    model.load(args.load)

    if not args.no_elbo_test:
        # Test the perplexity on test set
        perplexity = model.test_perplexity()
        print("Test perplexity: ", perplexity)

        marginal_perplexity = np.exp(model.test_on_dataset(tensor=model.marginal_log_perplexity))
        print("Marginal test perplexity: ", marginal_perplexity)
        # Print the top words with max probability of the topics
        if args.print_topic_file:
            with open(args.print_topic_file, "w") as f:
                model.print_top_words(file=f, use_tensor=args.print_all)
        else:
            # model.print_top_words(use_tensor=True)
            model.print_top_words(use_tensor=args.print_all)

    for test_module_name in args.test_module:
        test_module = available_test_modules[test_module_name]
        print("Run test module: {}".format(test_module_name))
        test_module.do_test(model, reader, args)

if __name__ == "__main__":
    main()
