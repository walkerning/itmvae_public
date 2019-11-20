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

class ImageVAE(VAE):
    """
    VAE for image gen model.
    """

    def build_rec_loss(self):
        # self.w_dist += 1e-10
        self.w_dist_3d = tf.reshape(self.w_dist, [self.cfg["MC_samples"], -1, self.cfg["vocab_dim"]])
        if self.train_cfg["image_loss_type"] == "binary":
            self.rec_loss = - tf.reduce_sum(self.x * tf.log(self.w_dist_3d) + (1 - self.x) * tf.log(1 - self.w_dist_3d), axis=-1)
        elif self.train_cfg["image_loss_type"] == "linear":
            self.rec_loss = 0.5 * tf.reduce_sum(tf.square(self.x - self.w_dist_3d), axis=-1)
        self.batch_rec_loss = tf.reduce_mean(self.rec_loss)
        tf.summary.scalar("batch reconstruction loss", self.batch_rec_loss)

    def _init_loss(self):
        pass

    @property
    def _test_tensor(self):
        return self.loss

    @property
    def _test_tensor_name(self):
        return "ELBO"

    @property
    def _topic_prop_tensor(self):
        return self.z_3d

    def _get_collapsed_word_dist(self, topic, test=False):
        w_dist = tf.layers.dense(topic, self.cfg["vocab_dim"], activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.zeros_initializer(), name="gen_layer_output")
        if self.cfg["recon_transfer_fct"]:
            w_dist = getattr(tf.nn, self.cfg["recon_transfer_fct"])(w_dist) + 1e-10
        return w_dist
