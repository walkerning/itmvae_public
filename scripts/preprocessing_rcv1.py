# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import argparse
import cPickle
import numpy as np
from scipy import io

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = "../../rcv2_data"
target_path = os.path.join(proj_root, "vae_topicmodel/datasets/", "rcv1_v2")
if not os.path.isdir(target_path):
    os.mkdir(target_path)

data_v = io.loadmat(os.path.join(data_path, "rcv2_data.mat"))
word_count = 10000
vocab = {data_v["words_10000"][i][0][0]:i for i in range(word_count)}
cPickle.dump(vocab, open(os.path.join(target_path, "vocab.pkl"), "w"))

for split in ["train", "valid", "test"]:
    fname = os.path.join(data_path, split + ".txt")
    with open(fname, "r") as f:
        records = [l.strip().split()[1:] for l in f.read().strip().split("\n")]
        # original index stored is 1-based
        records = [[(int(it.split(":")[0]) - 1, int(it.split(":")[1])) for it in l] for l in records]
    print("Split {}: {} samples.".format(split, len(records)))
    np.save(os.path.join(target_path, "{}.txt.npy".format(split)), records)
