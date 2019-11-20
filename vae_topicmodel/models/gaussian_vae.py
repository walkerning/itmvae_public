# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from .vae import VAE
from vae_topicmodel.utils import call_once

class GaussianVAE(VAE):
    _default_cfg = {
        # network structure
        "inference_net_structure": [100, 100],
        "gen_net_structure": [],
        "transfer_fct": "softplus",
        "recon_transfer_fct": None,
        "use_residual": False,
        "softmax_topic_vector": True,
        "dropout_topic_vector": False,
        "dropout_inference_out": True,
        "topic_role": "mixture", # could be one of `mixture`, `sage`
        "beta_regularizer": None, # if `sage` is used
        "sage_use_bias": False, # `sage`
        "sage_bias_init": False, # Use log(vocab occur) as bias initializer

        # regularizer configs
        "inference_regularizer": None,

        # batch norm configs
        "batch_norm_gen_net": False,
        "batch_norm_inference_net": False,
        "batch_norm_variational_param": True,
        "batch_norm_logits_wdist": True, # `sage`
        "batch_norm_beta": True, # `mixture`
        "batch_renorm": False,

        # initialize configs
        "logvar_bias_init": -4,
        "logvar_weight_init": 0,

        # stochastic layer
        "MC_samples": 1,

        # prior configs
        "prior_alpha": None,

        # summary
        "summary_dir": None,

        # centering inference net input
        "centering_input": False,

        "decompose_beta_dim": None,
        "beta_hidden_topic_dim": [],
        "beta_hidden_word_dim": [],

        "encoder_log_input": False
    }
    _cfg_handlers = {
        "transfer_fct": lambda name: getattr(tf.nn, name),
        "logvar_weight_init": lambda name: tf.contrib.layers.xavier_initializer() if name == "xavier" else tf.constant_initializer(name)
    }

    def _gaussian_log_pdf(self, x, mu, sigma):
        return -0.5 * (self.topic_dim * tf.cast(tf.log(2 * np.floatX(np.pi)), tf.floatX) + tf.reduce_sum(tf.log(sigma), axis=-1) + tf.reduce_sum(tf.square(x - mu) / sigma, axis=-1))

    @property
    @call_once
    def log_prior_pdf(self):
        return self._gaussian_log_pdf(self.z_3d, self.prior_mu, self.prior_var)

    @property
    @call_once
    def log_posterior_pdf(self):
        return self._gaussian_log_pdf(self.z_3d, self.z_mean, self.z_var)

    def sample_from_prior(self, num):
        eps = np.floatX(np.random.normal(size=(num, self.topic_dim)))
        return eps * np.sqrt(self.prior_var) + self.prior_mu

    def build_stochastic_layer(self, layer):
        self.z_mean = tf.layers.dense(layer, self.topic_dim, activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
        self.z_logvar = tf.layers.dense(layer, self.topic_dim, activation=None,
                                        kernel_initializer=self.cfg["logvar_weight_init"], bias_initializer=tf.constant_initializer(self.cfg["logvar_bias_init"]))
        if self.cfg["batch_norm_variational_param"]:
            self.z_mean = tf.contrib.layers.batch_norm(self.z_mean, is_training=self.training_placeholder)
            self.z_logvar = tf.contrib.layers.batch_norm(self.z_logvar, is_training=self.training_placeholder)
        # Reparametrization stochastic sampling layer
        eps = tf.random_normal((self.cfg["MC_samples"], tf.shape(self.x)[0], self.topic_dim), 0, 1, dtype=tf.floatX)
        self.z_var = tf.exp(self.z_logvar)
        self.z_3d = self.z_mean + tf.sqrt(self.z_var) * eps
        self.z = tf.reshape(self.z_3d, [-1, self.topic_dim])
        return self.z

    def build_kl_loss(self):
        # The prior.
        if self.cfg["prior_alpha"] is None:
            self.prior_mu = np.zeros([self.topic_dim], dtype=np.floatX)
            self.prior_var = np.ones([self.topic_dim], dtype=np.floatX)
        else:
            # For approximation of symmetric dirichlet prior in AVITM paper.
            self.prior_mu = tf.constant(np.zeros((1, self.topic_dim)).astype(np.floatX))
            self.prior_var = (1 - 1. / self.topic_dim) / self.cfg["prior_alpha"] * np.ones((1, self.topic_dim)).astype(np.floatX)

        self.KL_loss_3d = 0.5 * (self.z_var / self.prior_var +\
                                 tf.square(self.prior_mu - self.z_mean) / self.prior_var - 1 +\
                                 tf.log(self.prior_var) - self.z_logvar)
        self.KL_loss = tf.reduce_sum(self.KL_loss_3d, axis=-1)
        # self.KL_loss = 0.5 * (tf.reduce_sum(self.z_var / self.prior_var, axis=-1) + \
        #                       tf.reduce_sum(tf.square(self.prior_mu - self.z_mean) / self.prior_var, axis=-1) - \
        #                       self.topic_dim + tf.reduce_sum(tf.log(self.prior_var) - self.z_logvar, axis=-1))

        self.batch_kl_loss = tf.reduce_mean(self.KL_loss)
