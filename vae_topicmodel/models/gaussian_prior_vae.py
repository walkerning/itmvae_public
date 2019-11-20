# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from .gaussian_vae import GaussianVAE
from vae_topicmodel.utils import call_once

class GaussianPriorVAE(GaussianVAE):
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

        "encoder_log_input": False,

        "prior_mu": 0.0,
        "prior_lambda": 1.0,
        "prior_s1": 1.0,
        "prior_s2": 1.0,

        "trainable_post_mu": True,
        "trainable_post_lambda": True,
        "trainable_post_gamma1": True,
        "trainable_post_gamma2": True,

        "integral_out": False
    }
    _cfg_handlers = {
        "transfer_fct": lambda name: getattr(tf.nn, name),
        "logvar_weight_init": lambda name: tf.contrib.layers.xavier_initializer() if name == "xavier" else tf.constant_initializer(name)
    }

    @property
    @call_once
    def log_prior_pdf(self):
        K = self.topic_dim
        new_s1 = np.floatX(self.prior_s1 + K / 2)
        z_mean = tf.reduce_mean(self.z_3d, axis=-1, keep_dims=True)
        new_s2 = self.prior_s2 + tf.reduce_sum(tf.square(self.z_3d - z_mean), axis=-1) / 2 \
                 + self.prior_lambda * K / (self.prior_lambda + K) / 2 * tf.square(z_mean - self.prior_mu)
        return tf.lgamma(new_s1) - tf.lgamma(self.prior_s1) + np.floatX(self.prior_s1 * np.log(self.prior_s2)) - new_s1 * tf.log(new_s2) \
            - tf.cast(0.5 * tf.log(1 + K / self.prior_lambda), tf.floatX)

    def sample_from_prior(self, num):
        # lambda ~ Gamma(s_1, s_2)
        lambdas = np.random.gamma(scale=self.prior_s1, shape=1./self.prior_s2, size=(num,))
        # mu ~ Normal(mu_p, 1 / (lambda_p * lambda))
        mus = np.random.normal(loc=self.prior_mu, scale=np.sqrt(1./(self.prior_lambda * lambdas)))
        # x ~ Normal(mu, 1 / lambda)
        return np.floatX(np.random.normal(mus, np.sqrt(1. / lambdas), size=(self.topic_dim, num)).T)

    def build_kl_loss(self):
        self.prior_mu = np.floatX(self.cfg["prior_mu"])
        self.prior_lambda = np.floatX(self.cfg["prior_lambda"])
        self.prior_s1 = np.floatX(self.cfg["prior_s1"])
        self.prior_s2 = np.floatX(self.cfg["prior_s2"])

        self.post_mu = tf.get_variable("sgd_post_mu", shape=[], dtype=tf.floatX,
                                       initializer=tf.constant_initializer(self.prior_mu),
                                       trainable=self.cfg["trainable_post_mu"])
        self.log_post_lambda = tf.get_variable("sgd_log_post_lambda", shape=[], dtype=tf.floatX,
                                               initializer=tf.constant_initializer(np.log(self.prior_lambda)),
                                               trainable=self.cfg["trainable_post_lambda"])
        self.invsp_post_gamma1 = tf.get_variable("sgd_invsp_post_gamma1", shape=[], dtype=tf.floatX,
                                                 initializer=tf.constant_initializer(np.log(np.exp(self.prior_s1) - 1)),
                                                 trainable=self.cfg["trainable_post_gamma1"])
        self.invsp_post_gamma2 = tf.get_variable("sgd_invsp_post_gamma2", shape=[], dtype=tf.floatX,
                                                 initializer=tf.constant_initializer(np.log(np.exp(self.prior_s2) - 1)),
                                                 trainable=self.cfg["trainable_post_gamma2"])

        self.post_lambda = tf.exp(self.log_post_lambda)
        self.post_gamma1 = tf.nn.softplus(self.invsp_post_gamma1)
        self.post_gamma2 = tf.nn.softplus(self.invsp_post_gamma2)

        gamma_ratio = self.post_gamma1 / self.post_gamma2

        neg_log_entro = - self.topic_dim / 2 * (np.log(2 * np.pi) + 1) - tf.reduce_sum(self.z_logvar, axis=-1) / 2
        neg_log_entro_3d = - 1 / 2 * (np.log(2 * np.pi) + 1) - self.z_logvar / 2

        kl_normal_gamma = tf.lgamma(self.prior_s1) - tf.lgamma(self.post_gamma1) - self.prior_s1 * tf.log(self.prior_s2 / self.post_gamma2) \
                                                 - (np.log(self.prior_lambda) - tf.log(self.post_lambda)) / 2 \
                                                 - tf.digamma(self.post_gamma1) * (self.prior_s1 - self.post_gamma1) \
                                                 + gamma_ratio * self.prior_s2 - self.post_gamma1 - 0.5 \
                                                 + self.prior_lambda / self.post_lambda / 2 \
                                                 + self.prior_lambda * gamma_ratio / 2 * (self.prior_mu - self.post_mu) ** 2

        neg_Epz_q_z_u_lambda = self.topic_dim / 2 * (tf.log(2 * np.pi * self.post_gamma2) + 1 / self.post_lambda \
                                                     + self.post_mu ** 2 * gamma_ratio - tf.digamma(self.post_gamma1)) \
                                                 + gamma_ratio / 2 * tf.reduce_sum(tf.square(self.z_mean) + self.z_var - 2 * self.z_mean * self.post_mu, axis=-1)
        neg_Epz_q_z_u_lambda_3d = 0.5 * (tf.log(2 * np.pi * self.post_gamma2) + 1 / self.post_lambda \
                                         + self.post_mu ** 2 * gamma_ratio - tf.digamma(self.post_gamma1)) \
            + gamma_ratio / 2 * (tf.square(self.z_mean) + self.z_var - 2 * self.z_mean * self.post_mu)

        self.KL_loss_3d = neg_log_entro_3d + kl_normal_gamma + neg_Epz_q_z_u_lambda_3d

        self.KL_loss = neg_log_entro + kl_normal_gamma + neg_Epz_q_z_u_lambda
        self.batch_kl_loss = tf.reduce_mean(self.KL_loss)
