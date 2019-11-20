# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import itertools
import cPickle

import numpy as np
import tensorflow as tf
import scipy.sparse as sps

here = os.path.dirname(__file__)

class Reader(object):
  path_suffix = ".txt.npy"

  def __init__(self, dataset, load=("train", "valid", "test"), add_valid_to_train=False):
    self.dataset = dataset
    self.load_splits = load
    self.add_valid_to_train = add_valid_to_train
    if self.add_valid_to_train:
      print("Will use valid data for training, do not use for hold-out validation.")
    data_path = os.path.join(here, "datasets", dataset)
    if not os.path.exists(data_path):
      raise Exception("Dataset `{}` not found under path {}.".format(dataset,
                                                                     os.path.join(here, "datasets")))
    self.train_path = os.path.join(data_path, "train" + self.path_suffix)
    self.valid_path = os.path.join(data_path, "valid" + self.path_suffix)
    self.test_path = os.path.join(data_path, "test" + self.path_suffix)
    vocab_path = os.path.join(data_path, "vocab.pkl")
    paths = [vocab_path, self.train_path, self.valid_path, self.test_path]
    if not all(os.path.exists(p) for p in paths):
      raise Exception("Need " + ", ".join(["`{}`".format(p) for p in paths]) + " files."
                      "Write preprocessing script to generate these files.")

    self.vocab = cPickle.load(open(vocab_path, "r"))
    self.vocab_dim = len(self.vocab)
    self.reverse_vocab = {v:k for k, v in self.vocab.iteritems()}

    assert isinstance(load, (tuple, list)), "Argument load must be a list or a tuple. Receive a {} (`{}`)".format(type(load), load)
    for split in load:
      split_data = self._load_data(open(getattr(self, split + "_path"), "r"))
      good_indexes, split_data = zip(*[(i, data) for i, data in enumerate(split_data) if self.isgood(data)])
      setattr(self, split + "_data", split_data)
      setattr(self, split + "_good_indexes", good_indexes) # use to filter labels
      sz = len(split_data)
      setattr(self, split + "_size", sz)
      print("Loaded {} data. {} samples.".format(split, sz))
    if self.add_valid_to_train and "train" in self.load_splits:
      self.train_size = self.train_size + self.valid_size
    self._cache = {}

  def concat(self, dt1, dt2):
    return dt1 + dt2

  def _load_data(self, name):
    return np.load(name)

  def load_label_from_type(self, data_type):
    assert data_type in {"train", "valid", "test"}
    if hasattr(self, data_type + "_labels"):
      return getattr(self, data_type + "_labels")
    data_path = os.path.join(here, "datasets", self.dataset)
    good_indexes = np.array(getattr(self, data_type + "_good_indexes"), dtype=np.int32)
    labels = np.squeeze(np.load(os.path.join(data_path, data_type + ".label.npy")))[good_indexes]
    if self.add_valid_to_train and data_type == "train":
      labels = np.concatenate((labels, self.load_label_from_type("valid")))
    setattr(self, data_type + "_labels", labels)
    return labels

  def filter_data_labels(self, keep_labels):
    """
    Only keep the data with label in the `keep_labels`.

    Arguments:
    keep_labels: iterable of labels.
    """
    keep_labels = set(keep_labels)
    for split in self.load_splits:
      data_path = os.path.join(here, "datasets", self.dataset)
      good_indexes = np.array(getattr(self, split + "_good_indexes"), dtype=np.int32)
      split_labels = np.squeeze(np.load(os.path.join(data_path, split + ".label.npy")))[good_indexes]
      inds, filtered_labels = zip(*[(ind, label) for ind, label in enumerate(split_labels) if label in keep_labels])
      ori_data = getattr(self, split + "_data")
      setattr(self, split + "_data", [ori_data[ind] for ind in inds])
      sz = len(inds)
      setattr(self, split + "_size", sz)
      setattr(self, split + "_labels", filtered_labels)
      print("Filtered {} data using labels {}: {} samples.".format(split, keep_labels, sz))

  def isgood(self, data):
    return data != []

  def get_data_from_type(self, data_type):
    if data_type == "train":
      raw_data = self.train_data
      if self.add_valid_to_train:
        raw_data = self.concat(self.train_data, self.valid_data)
    elif data_type == "valid":
      raw_data = self.valid_data
    elif data_type == "test":
      raw_data = self.test_data
    else:
      raise Exception(" [!] Unkown data type %s: %s" % data_type)

    return raw_data

  def get_parsed_data_from_type(self, data_type, func=lambda x: x):
    """
    ATTENTION: All one-hot data will be put in memory.
    """
    # if (data_type, func) not in self._cache:
    #   datas = self.get_data_from_type(data_type)
    #   self._cache[(data_type, func)] = func(np.array([self.onehot(data) for data in datas if self.isgood(data)]))
    # return self._cache[(data_type, func)]
    datas = self.get_data_from_type(data_type)
    return func(np.array([self.onehot(data) for data in datas if self.isgood(data)]))

  def mean(self, data_type="train", func=lambda x: x):
    # FIXME: what if the dataset is too big to fit in memory... Better calculate mean in advance.
    return np.mean(self.get_parsed_data_from_type(data_type, func=func), axis=0)

  def iterator_one_pass(self, batch_size, data_type="train", random=False, shuffle=False, func=lambda x: x):
    """
    Go through the dataset one pass and stop. The final batch can be different length.
    By default, not random, used for test.
    """
    data_sz = getattr(self, data_type + "_size")
    steps = data_sz // batch_size
    residual = data_sz - steps * batch_size
    gen = self.iterator(batch_size, data_type=data_type, random=random, shuffle=False, func=func)
    for _ in range(steps):
      yield gen.next()
    if residual > 0:
      yield gen.next()[:residual]

  def iterator(self, batch_size, data_type="train", random=True, shuffle=False, func=lambda x: x):
    def _iterator(batch_size, data_type="train", random=True, shuffle=False):
      raw_data = self.get_data_from_type(data_type)
      if not random:
        # Sequential
        iterator = itertools.cycle((self.onehot(data) for data in raw_data if self.isgood(data)))
        while 1:
          yield np.array([iterator.next() for _ in range(batch_size)])
      else:
        rng = np.random.RandomState(10)
        data_sz = getattr(self, data_type + "_size")
        if not shuffle:
          # randomly pick data
          while 1:
            ixs = rng.randint(data_sz, size=batch_size)
            yield np.array([self.onehot(raw_data[i]) for i in ixs])
        else:
          # shuffle
          idxes = np.arange(data_sz)
          steps_per_epoch = data_sz // batch_size
          step = 0
          while 1:
            if step % steps_per_epoch == 0:
              rng.shuffle(idxes)
              step = 0
            yield np.array([self.onehot(raw_data[i]) for i in idxes[step * batch_size: (step + 1) * batch_size]])
            step += 1
    gen = _iterator(batch_size, data_type, random, shuffle)
    while 1:
      x = gen.next()
      yield func(x)

  registry = {
    "index": "IndexReader",
    "index_sparse": "IndexSparseReader",
    "sparse": "SparseReader"
  }

  @classmethod
  def get_reader_cls(cls, name):
    print("Use reader type {}.".format(name))
    thismodule = sys.modules[__name__]
    return getattr(thismodule, cls.registry[name])


class IndexReader(Reader):
  def onehot(self, data, min_length=None):
    if min_length == None:
      min_length = self.vocab_dim
    return np.bincount(data, minlength=min_length)

  def to_sparse_matrix(self, data_type):
    index_data = self.get_data_from_type(data_type)
    mat = sps.csr_matrix(np.zeros((0, self.vocab_dim), dtype=np.int32))
    for d in index_data:
      mat = sps.vstack((mat, sps.csr_matrix(np.bincount(d, minlength=self.vocab_dim))))
    return mat


class IndexSparseReader(Reader):
  def onehot(self, data, min_length=None):
    if min_length == None:
      min_length = self.vocab_dim
    x = np.zeros(min_length, dtype=np.int32)
    for ass in data:
      x[ass[0]] = ass[1]
    return x

  def to_sparse_matrix(self, data_type):
    indexsparse_data = self.get_data_from_type(data_type)
    ij, values = zip(*[((i, di[0]), di[1]) for di in d for i, d in enumerate(indexsparse_data)])
    mat = sps.csr_matrix((np.array(values, dtype=np.int32), np.array(ij).transpose()), shape=(len(indexsparse_data), self.vocab_dim))
    return mat

class SparseReader(Reader):
  path_suffix = "_data.npz"
  def onehot(self, data, min_length=None):
    return np.squeeze(data.toarray())

  def isgood(self, data):
    return data.sum() > 0

  def _load_data(self, name):
    mat = np.load(name)
    return sps.csr_matrix((mat["data"], mat["indices"], mat["indptr"]), shape=tuple(list(mat["shape"])))

  def to_sparse_matrix(self, data_type):
    return self.get_data_from_type(data_type)
