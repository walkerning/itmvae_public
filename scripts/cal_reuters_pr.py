# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import argparse
import scipy.sparse as sps
import numpy as np
from BoW_20News_probout import GetPrecisionMatrix

parser = argparse.ArgumentParser()
parser.add_argument("load", help="Load the tensors from path.")
parser.add_argument("--save", help="save results to", default=None)
parser.add_argument("--dataset", default="reuters_uai")
args = parser.parse_args()

dataset = args.dataset
if args.save is None:
    args.save = os.path.join(args.load, "pr.npz")
print("loading labels")
label_path = os.path.join("vae_topicmodel/datasets", dataset)
train_labels = np.load(os.path.join(label_path, "train.label.npy"))
if dataset == "reuters_uai":
    train_labels = sps.csr_matrix((train_labels["data"], train_labels["indices"], train_labels["indptr"]), shape=tuple(list(train_labels["shape"])))
valid_labels = np.load(os.path.join(label_path, "valid.label.npy"))
if dataset == "reuters_uai":
    valid_labels = sps.csr_matrix((valid_labels["data"], valid_labels["indices"], valid_labels["indptr"]), shape=tuple(list(valid_labels["shape"])))
if dataset == "reuters_uai":
    train_labels = sps.vstack((train_labels, valid_labels)).toarray()
else:
    train_labels = np.squeeze(np.concatenate((train_labels, valid_labels)))

test_labels = np.load(os.path.join(label_path, "test.label.npy"))
if dataset == "reuters_uai":
    test_labels = sps.csr_matrix((test_labels["data"], test_labels["indices"], test_labels["indptr"]), shape=tuple(list(test_labels["shape"]))).toarray()
else:
    test_labels = np.squeeze(test_labels)
print(train_labels.shape, test_labels.shape)

print("loading topics")
train_topics = np.load(os.path.join(args.load, "train.npy"))
test_topics = np.load(os.path.join(args.load, "test.npy"))
assert len(train_topics) == train_labels.shape[0]
assert len(test_topics) == test_labels.shape[0]


print("Calculating precision, recalls matrices.")
precision, grid_recall, ap, pq_ap, pq_prec = GetPrecisionMatrix(
    train_topics, train_labels, test_topics, test_labels,
    numqueries='all', numdatabase='all', parallelize=dataset == "reuters_uai", multilabel=dataset == "reuters_uai")

prfile = os.path.expanduser(args.save)
print("Writing precision, recall matrices to {}".format(prfile))
np.savez(prfile, precision=precision, recall=grid_recall)
