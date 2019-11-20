# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import yaml
import argparse

import tensorflow as tf

from vae_topicmodel.models import Model
from vae_topicmodel.reader import Reader

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
    parser.add_argument("--reader-type", default="index", choices=Reader.registry.keys())
    parser.add_argument("--keep-labels", default=None,
                        help="Comma-split list of label indexes that will be kept(eg. --keep-labels 0,1,2,3). By default, all labels will be kept.")
    # 用mc samples还是期望... AVITM里用的是sample.
    parser.add_argument("--mc-samples", default=20, type=int)

    args = parser.parse_args()
    args.load = os.path.abspath(args.load)
    print("The model will be loaded from ", args.load)

    # Construct dataset reader, test only
    reader = Reader.get_reader_cls(args.reader_type)(args.dataset)#, #load=("test",))
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
    cfg["MC_samples"] = args.mc_samples
    cfg["reader"] = reader
    cfg["vocab_dim"] = reader.vocab_dim

    model = Model.init_model_from_cfg(cfg, train_cfg)
    model.init(test_only=True)
    model.load(args.load)

    self.onehot(reader.get_data_from_type("test")[0])
    # # Test the perplexity on test set
    # perplexity = model.test_perplexity()
    # print("Test perplexity: ", perplexity)

    # # Print the top words with max probability of the topics
    # if args.print_topic_file:
    #     with open(args.print_topic_file, "w") as f:
    #         model.print_top_words(file=f)
    # else:
    #     model.print_top_words()

if __name__ == "__main__":
    main()
