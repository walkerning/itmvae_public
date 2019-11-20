# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import sys
import copy
import cPickle
import tempfile
from datetime import datetime

import numpy as np
import tensorflow as tf

from .vae import VAE

def _log_input(x):
    return np.log(1 + x)

class BoWVAE(VAE):
    """
    VAE for BoW topic model.
    """

    def build_rec_loss(self):
        # Let's smooth the multinomial parameters.
        self.w_dist += 1e-10
        self.w_dist_3d = tf.reshape(self.w_dist, [self.cfg["MC_samples"], -1, self.cfg["vocab_dim"]])
        self.rec_loss = - tf.reduce_sum(self.x * tf.log(self.w_dist_3d), axis=-1)
        self.batch_rec_loss = tf.reduce_mean(self.rec_loss)
        tf.summary.scalar("batch reconstruction loss", self.batch_rec_loss)

    def _init_loss(self):
        self.log_perplexity = tf.reduce_mean((self.rec_loss + self.KL_loss) / tf.reduce_sum(self.x, axis=-1))
        log_weighted_p = -self.rec_loss + self.log_prior_pdf - self.log_posterior_pdf

        base = tf.reduce_max(log_weighted_p, axis=0, keep_dims=True)
        log_weighted_p = log_weighted_p - base
        weighted_p = tf.exp(log_weighted_p)
        p_xi = tf.reduce_mean(weighted_p, axis=0)
        
        self.marginal_log_perplexity = -tf.reduce_mean((tf.log(p_xi) + base) / tf.reduce_sum(self.x, axis=-1))
        tf.summary.scalar("batch log perplexity", self.log_perplexity)

    def test_perplexity(self, data_type="test"):
        return np.exp(self.test_on_dataset(data_type))

    @property
    def _test_tensor(self):
        return self.log_perplexity

    @property
    def _test_tensor_name(self):
        return "log perplexity"

    @property
    def _topic_components_tensor(self):
        return self.beta

    @property
    def _topic_prop_tensor(self):
        return self.theta

    def _get_collapsed_word_dist(self, topic, test=False):
        topic_role = self.cfg["topic_role"]
        if self.cfg["decompose_beta_dim"] is None:
            self.beta = tf.get_variable("beta", [self.topic_dim, self.vocab_dim], tf.floatX,
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        regularizer=(tf.contrib.layers.l1_regularizer(scale=self.cfg["beta_regularizer"])
                                                     if topic_role == "sage" and self.cfg["beta_regularizer"] is not None else None))
        else:
            print("Decompose beta embedding dim: ", self.cfg["decompose_beta_dim"])
            beta_word = tf.get_variable("beta_word", [self.cfg["decompose_beta_dim"], self.vocab_dim], tf.floatX,
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             regularizer=(tf.contrib.layers.l2_regularizer(scale=self.cfg["beta_regularizer"]) if topic_role == "sage" and self.cfg["beta_regularizer"] is not None else None))
            self.beta_topic_begin = beta_topic = tf.get_variable("beta_topic", [self.topic_dim, self.cfg["decompose_beta_dim"]], tf.floatX,
                                              initializer=tf.contrib.layers.xavier_initializer(),
                                              regularizer=(tf.contrib.layers.l2_regularizer(scale=self.cfg["beta_regularizer"])
                                                           if topic_role == "sage" and self.cfg["beta_regularizer"] is not None else None))
            beta_word = tf.transpose(beta_word)
	    if self.cfg.has_key("beta_hidden_word_dim"):
                beta_hidden_word_dim = self.cfg["beta_hidden_word_dim"]
                for i, dim in enumerate(beta_hidden_word_dim):
                    beta_word = tf.layers.dense(beta_word, dim, activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            bias_initializer=tf.zeros_initializer(), name="beta_word_layer_{}".format(i+1),
					    kernel_regularizer=(tf.contrib.layers.l2_regularizer(scale=self.cfg["beta_regularizer"])
                                                     if topic_role == "sage" and self.cfg["beta_regularizer"] is not None else None))
	    if self.cfg.has_key("beta_hidden_topic_dim"):
                beta_hidden_topic_dim = self.cfg["beta_hidden_topic_dim"]
                for i, dim in enumerate(beta_hidden_topic_dim):
                    beta_topic = tf.layers.dense(beta_topic, dim, activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            bias_initializer=tf.zeros_initializer(), name="beta_topic_layer_{}".format(i+1),
					    kernel_regularizer=(tf.contrib.layers.l2_regularizer(scale=self.cfg["beta_regularizer"])
                                                     if topic_role == "sage" and self.cfg["beta_regularizer"] is not None else None))
            self.beta_word = tf.transpose(beta_word)
            self.beta_topic = beta_topic
            self.beta = tf.matmul(self.beta_topic, self.beta_word)

        if topic_role == "mixture":
            if self.cfg["batch_norm_beta"]:
                beta_bn = tf.contrib.layers.batch_norm(self.beta, is_training=(not test) and self.training_placeholder, scope="batch_norm_beta", renorm=self.cfg.get("batch_renorm", False))
            else:
                beta_bn = self.beta
            return tf.matmul(topic, tf.nn.softmax(beta_bn))
        else: # sage
            if self.cfg.get("normalize_topic_diff_scale", False):
                topic_diff_scale = tf.sqrt(tf.reduce_mean(tf.square(self.beta - tf.reduce_mean(self.beta, axis=-1, keep_dims=True)), axis=-1) + 1e-8) # std
                scale_std = tf.sqrt(tf.reduce_mean(tf.square(topic_diff_scale - tf.reduce_mean(topic_diff_scale))) + 1e-8) # std of std
                self.beta = self.beta / tf.stop_gradient(scale_std)
            elif self.cfg.get("log_beta", False):
                self.pre_log_beta = self.beta
                sign_mask = tf.stop_gradient(2 * tf.cast(self.beta >= 0, tf.float32) - 1)
                self.beta = sign_mask * tf.log(1 + sign_mask * self.beta)
            logits_w_dist = tf.matmul(topic, self.beta)
            if self.cfg["sage_use_bias"]:
                if self.cfg["sage_bias_init"]:
                    # TODO: seems `log_input` configuration should be passed to reader rather
                    # than used in model... Very ugly..
                    def _log_input(x):
                        return np.log(1 + x)
                    # The bias scale should have influence too... it should be
                    # ajust according to the leraning rate and the initialization of
                    # `beta`
                    bias_init = np.log(self.reader.mean(func=_log_input if self.train_cfg["log_input"] else lambda x: x) + 1e-10)
                    self.sage_bias = tf.get_variable("gen_sage_background_bias", [self.vocab_dim], tf.floatX,
                                                     initializer=tf.constant_initializer(bias_init))
                else:
                    self.sage_bias = tf.get_variable("gen_sage_background_bias", [self.vocab_dim], tf.floatX)
                logits_w_dist = logits_w_dist + self.sage_bias
            if self.cfg["batch_norm_logits_wdist"]:
                logits_w_dist = tf.contrib.layers.batch_norm(logits_w_dist, is_training=(not test) and self.training_placeholder, scope="batch_norm_logits_wdist", renorm=self.cfg.get("batch_renorm", False))
            return tf.nn.softmax(logits_w_dist)
