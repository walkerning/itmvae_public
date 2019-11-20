# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import copy
import cPickle
from datetime import datetime

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from base import Model

class DMFVI(Model):
    MODEL_NAME = "dmfvi"
    _default_cfg = {
        "learning_method": "batch",
        "max_iter": 10,
        "batch_size": 128,
        "perp_tol": 0.1,
        "evaluate_every": 10
    }

    def __init__(self, cfg, train_cfg):
        super(DMFVI, self).__init__(cfg, train_cfg)

        self.cfg = copy.deepcopy(self._default_cfg)
        self.cfg.update(cfg)
        model_kwargs = {k: v for k, v in self.cfg.iteritems() if k in self._default_cfg}
        self.model = LatentDirichletAllocation(n_components=self.topic_dim, verbose=2, **model_kwargs)
        print("DMFVI: Use model configration:\n{}".format("\n".join("\t{:30}: {}".format(k, v) for k, v in sorted(model_kwargs.iteritems(), key=lambda item: item[0]))))

    def perplexity(self, x):
        return self.model.perplexity(np.array(x))

    def topic_prop(self, x):
        return self.model.transform(x)

    @property
    def topic_components(self):
        return self.model.components_

    def train(self):
        train_data = self.reader.get_data_from_type("train")
        self.model.fit(np.array([self.reader.onehot(data) for data in train_data if data != []]))
        print ("{}: trained for {} epochs; {} EM iterations.".format(datetime.now(), self.model.n_iter_, self.model.n_batch_iter_))

    def save(self, path):
        cPickle.dump(self.model, open(path, "w"))

    def load(self, path):
        self.model = cPickle.load(open(path, "r"))
