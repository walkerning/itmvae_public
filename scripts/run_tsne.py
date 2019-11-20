# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import yaml
import random
import argparse
import subprocess

import numpy as np
from sklearn.manifold import TSNE
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vae_topicmodel.models import Model
from vae_topicmodel.reader import Reader

parser = argparse.ArgumentParser()
parser.add_argument("load", help="Load the model from path.")
parser.add_argument("save", help="Save the tsne fig to path.")
parser.add_argument("--cfg-file", default=None, help="The config file used.")
parser.add_argument("--train-cfg-file", default=None, help="The training config file, only test configurations will be used.")
parser.add_argument("--model", default=None, choices=Model.available_models,
                    help="The model used, override what is defined in cfg file.")

parser.add_argument("--topic-dim", default=None, type=int,
                    help="The topic dimension, override what is defined in cfg file.")
parser.add_argument("--dataset", metavar="DATASET", default="20news",
                    help="The name of the dataset used,"
                    "the dataset should be put under directory `<project root>/datasets/<DATASET>`.")
parser.add_argument("--seed", default=12345, type=int,
                    help="The random seed used for random libs.")
parser.add_argument("--reader-type", default="index", choices=Reader.registry.keys())
parser.add_argument("--effective-dim", type=int, default=None, help="By default, the effective dim is choosed as topic dim") # 先取前这么多个, 至少从assignment看起来没问题.
parser.add_argument("--sample-num", type=int, default=None, help="By default, use the whole test set.")

args = parser.parse_args()
print("The model will be loaded from ", args.load)

if args.seed:
    print("Set random seed to {}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

# Construct dataset reader, test only
reader = Reader.get_reader_cls(args.reader_type)(args.dataset, load=("test",))
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

if args.effective_dim is None:
    args.effective_dim = cfg["topic_dim"]

train_cfg = yaml.load(open(args.train_cfg_file, "r").read()) if args.train_cfg_file else {}

cfg["reader"] = reader
cfg["vocab_dim"] = reader.vocab_dim

model = Model.init_model_from_cfg(cfg, train_cfg)
model.init(test_only=True)
model.load(args.load)
labels = reader.load_label_from_type("test")[:args.sample_num]
print("Calculating topics...")
topics = model.topic_prop(reader.get_parsed_data_from_type("test")[:args.sample_num])[:, :args.effective_dim]

print("Start fitting TSNE... sample num: {}; effective dim: {}".format(args.sample_num, args.effective_dim))
tsne = TSNE(n_components=2, random_state=0)
embeding = tsne.fit_transform(topics)
plt.figure(figsize=(6, 6))
plt.scatter(embeding[:, 0], embeding[:, 1], c=labels, cmap='jet')
plt.colorbar()
plt.savefig(args.save)
