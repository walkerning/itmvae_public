# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import sys
import abc
import six
import yaml

import numpy as np

class _ModelMeta(abc.ABCMeta):
    _model_registry = {}
    def __new__(mcls, name, bases, attrs):
        reg_name = attrs.get("MODEL_NAME", None)
        cls = super(_ModelMeta, mcls).__new__(mcls, name, bases, attrs)
        if reg_name is not None:
            mcls._model_registry[reg_name] = cls
            print("Registering model {}, implementation class: {}".format(reg_name, name))
        return cls

    @property
    def available_models(cls):
        return cls._model_registry.keys()

@six.add_metaclass(_ModelMeta)
class Model(object):
    def __init__(self, cfg, train_cfg):
        # Construct the reader
        self.reader = cfg["reader"]
        self.vocab_dim = cfg["vocab_dim"]
        self.topic_dim = cfg["topic_dim"]

    def test_perplexity(self, data_type="test"):
        """
        Likelihood per word(average in log space).
        This is not the same as the likelihood per doc.
        exp(-1. * log-likelihood per word)

        Here the perplexity is calculated across all test samples.
        In contrast, in AVITM code, the perplexity is calculated per test sample,
        and then caculate the mean.
        """
        test_data = self.reader.get_data_from_type(data_type)
        return self.perplexity(np.array([self.reader.onehot(data) for data in test_data if data != []]))

    def print_top_words(self, reverse_vocab=None, n_top_words=10, file=None, beta=None, use_tensor=False):
        beta = beta if beta is not None else (self.topic_components if not use_tensor else self.sess.run(self._topic_components_tensor))
        reverse_vocab = reverse_vocab or self.reader.reverse_vocab
        if file is None:
            print("---------------Printing the Topics------------------")
        for i in range(len(beta)):
            print(" ".join([reverse_vocab[j]
                            for j in beta[i].argsort()[:-n_top_words - 1:-1]]), file=file or sys.stdout)
        if file is None:
            print("---------------End of Topics------------------")

    ## Nonabstract/Abstract methods that can/should be override in subclasses
    def init(self, test_only=False):
        pass

    def perplexity(self, x):
        pass

    @abc.abstractmethod
    def topic_prop(self, x):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass

    @property
    def topic_components(self):
        pass

    ## Model loader class methods
    @classmethod
    def init_model_from_cfg(cls, cfg, train_cfg, model=None):
        if model is None:
            model = cfg["model"]
        return cls.get_model_cls(model)(cfg, train_cfg)

    @classmethod
    def get_model_cls(cls, name):
        imp_cls = cls._model_registry.get(name, None)
        if imp_cls is None:
            raise Exception("Model with name '{}' not registered.".format(name))
        return imp_cls
