# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import gzip
import cPickle

from .reader import Reader

here = os.path.dirname(__file__)

class ImageReader(Reader):
    def __init__(self, dataset, add_valid_to_train=False):
        self.add_valid_to_train = add_valid_to_train
        data_path = os.path.join(here, "datasets", dataset, "{}.pkl.gz".format(dataset))
        if not os.path.exists(data_path):
            data_path = os.path.join(here, "datasets", dataset, "{}.pkl".format(dataset))
            with open(data_path, "r") as f:
                trainset, validset, testset = cPickle.load(f)
        else:
            with gzip.open(data_path, "rb") as f:
                trainset, validset, testset = cPickle.load(f)

        self.train_data = trainset[0]
        self.train_labels = trainset[1]
        self.train_size = len(trainset[1])

        self.valid_data = validset[0]
        self.valid_labels = validset[1]
        self.valid_size = len(validset[1])

        self.test_data = testset[0]
        self.test_labels = testset[1]
        self.test_size = len(testset[1])

        self.vocab_dim = self.train_data.shape[1]

    def isgood(self, data):
        return True

    def onehot(self, data):
        return data

    def filter_data_labels(self, keep_labels):
        """
        Only keep the data with label in the `keep_labels`.

        Arguments:
        keep_labels: iterable of labels.
        """
        keep_labels = set(keep_labels)
        for split in ("train", "valid", "test"):
            data_path = os.path.join(here, "datasets", self.dataset)
            split_labels = getattr(self, split + "_labels")
            inds, filtered_labels = zip(*[(ind, label) for ind, label in enumerate(split_labels) if label in keep_labels])
            ori_data = getattr(self, split + "_data")
            setattr(self, split + "_data", [ori_data[ind] for ind in inds])
            sz = len(inds)
            setattr(self, split + "_size", sz)
            setattr(self, split + "_labels", filtered_labels)
            print("Filtered {} data using labels {}: {} samples.".format(split, keep_labels, sz))

    def load_label_from_type(self, data_type):
        assert data_type in {"train", "valid", "test"}
        return getattr(self, data_type + "_labels")
