# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import yaml
import argparse
import subprocess

import tensorflow as tf

from vae_topicmodel.models import Model
from vae_topicmodel.reader import Reader

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
parser.add_argument("--result-dir", default="./results/test_effective")
parser.add_argument("--context-size", default=20, type=int)
parser.add_argument("--indicator", default="assignment")
parser.add_argument("-t", "--threshold-lst", action="append", type=float)

args = parser.parse_args()
print("The model will be loaded from ", args.load)

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

train_cfg = yaml.load(open(args.train_cfg_file, "r").read()) if args.train_cfg_file else {}

cfg["reader"] = reader
cfg["vocab_dim"] = reader.vocab_dim
cfg["effective_indicator"] = args.indicator

model = Model.init_model_from_cfg(cfg, train_cfg)
model.init(test_only=True)
model.load(args.load)

effective_indicator = "assignment"
if args.threshold_lst:
    threshold_lst = args.threshold_lst
else:
    threshold_lst = [0, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]

print("Threshold list: ", threshold_lst)

for threshold in threshold_lst:
    model.set_threshold(threshold)
    resfile = os.path.join(args.result_dir, str(threshold) + ".txt")
    print("Threshold: {}".format(threshold))
    with open(resfile, "w") as f:
        model.print_top_words(file=f)
    subprocess.check_call("bash ./run_npmi.sh {} {} {} >/dev/null 2>&1 && tail -n 2 {}".format(resfile, args.dataset, args.context_size, resfile + ".oc"), shell=True)
