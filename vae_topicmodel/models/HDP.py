# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import copy

try:
    import gensim
except ImportError as e:
    print("Package gensim not available; HDP method ignored.")
else:
    from base import Model

    class HDP(Model):
        """
        Simple wrapper of gensim HDP model. Very slow, do not use this.
        """

        MODEL_NAME = "hdp"
        _default_cfg = {
            "T": 100,
            "K": 10,
            "gamma": 1,
            "alpha": 1,
            "epochs": 200
        }

        def __init__(self, cfg, train_cfg):
            super(HDP, self).__init__(cfg, train_cfg)

            self.cfg = copy.deepcopy(self._default_cfg)
            self.cfg.update(cfg)
            model_kwargs = {
                k: v for k, v in self.cfg.iteritems() if k in self._default_cfg}
            print("HDP: Use model configration:\n{}".format("\n".join("\t{:30}: {}".format(
                k, v) for k, v in sorted(model_kwargs.iteritems(), key=lambda item: item[0]))))
            self.train_corpus = gensim.matutils.Sparse2Corpus(
                self.reader.to_sparse_matrix("train"), documents_columns=False)
            self.test_corpus = gensim.matutils.Sparse2Corpus(
                self.reader.to_sparse_matrix("test"), documents_columns=False)
            print("Construct gensim corpus finished...")
            self.model = gensim.models.hdpmodel.HdpModel(self.train_corpus, {ind: word for word, ind in self.reader.vocab.iteritems()}, T=self.cfg["T"], K=self.cfg["K"],
                                                         gamma=self.cfg["gamma"], alpha=self.cfg["alpha"])
            print("Construct HDP model finished...")

        def test_perplexity(self, data_type="test"):
            assert data_type == "test"
            return self.model.evaluate_test_corpus(self.test_corpus)

        def train(self):
            score = self.model.evaluate_test_corpus(self.test_corpus)
            print("Start training... Start perplexity score: ", score)
            epochs = self.cfg["epochs"]
            for epoch in range(1, epochs+1):
                self.model.m_num_docs_processes = 0
                self.model.update(self.train_corpus)
                score = self.model.evaluate_test_corpus(self.test_corpus)
                print("Epoch {}: score: {}".format(epoch, score))

        def save(self, path):
            self.model.save(path)

        def load(self, path):
            self.model.load(path)

        def topic_prop(self, x):
            return self.model.get_topics()

        @property
        def topic_components(self):
            return self.model.get_topics()
