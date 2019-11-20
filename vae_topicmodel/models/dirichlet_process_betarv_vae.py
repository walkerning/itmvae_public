# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from .dirichlet_process_vae import DirichletProcessVAE, _beta_fn, smooth_log
from vae_topicmodel.utils import call_once

class DirichletProcessBetaRVVAE(DirichletProcessVAE):
    """
    Add gamma prior for `beta` parameter of the prior beta distribution.
    """
    _default_cfg = {
        # network structure
        "inference_net_structure": [100, 100],
        #"inference_net_structure": [500],
        "gen_net_structure": [],
        "transfer_fct": "softplus",
        "recon_transfer_fct": None,
        "use_residual": False,
        "softmax_topic_vector": False,
        "dropout_topic_vector": False,
        "dropout_inference_out": True,
        "topic_role": "mixture", # could be one of `mixture`, `sage`
        "beta_regularizer": None, # if `sage` is used
        "sage_use_bias": False, # `sage`
        "sage_bias_init": False,

        # batch norm configs
        "batch_norm_gen_net": False,
        "batch_norm_inference_net": False,
        "batch_norm_logits_wdist": True, # `sage`
        "batch_norm_beta": True, # `mixture`
        "batch_renorm": False,

        # initialize configs
        "logvar_bias_init": -4,
        "logvar_weight_init": 0,
        
        # prior configs
        "prior_alpha": 1.,
        # "prior_beta": 5.,

        # summary
        "summary_dir": None,

        # Stochastic layer
        "dirichlet_ab_fct": "softplus",
        "dirichlet_ab_use_bias": True,
        "MC_samples": 1,

        "centering_input": False,

        # stick breaking
        "effective_indicator": "assignment", # one of `average`, `assignment`
        "stick_epsilon": 0.05, # 95% stick coverage
        "effective_threshold": 0.02, # average reaches 2%... not a very good definition...
        "assignment_threshold": 0.005, # at least 0.5% document is assigned to this topic
        "num_kl_term": 15,

        "decompose_beta_dim": None,

        "b_init": 0,

        "encoder_log_input": False,
        "inference_regularizer": None,
        
        # Treat beta as r.v. prior gamma distribution
        # "prior_s1": 10.0, # shape parameter
        #"prior_s2": 1.0 # inverse scale parameter
        # "prior_s2": 0.25 # inverse scale parameter
        # "prior_s2": 1.0 # inverse scale parameter
        "prior_s1": 1, # shape parameter
        "prior_s2": 0.25, # inverse scale parameter

        # Use variational inference or integral out $beta$
        "integral_out": False
    }

    @property
    @call_once
    def log_prior_pdf(self):
        # For stick-breaking, prior_a = 1
        # \int \prod_i p(z_i | \beta) p(\beta) \diff{\beta}
        K = self.topic_dim - 1
        tmp = tf.reduce_sum(smooth_log(1 - self.vs), axis=-1)
        return tf.lgamma(np.floatX(self.prior_s1 + K)) - tf.lgamma(self.prior_s1) + self.prior_s1 * tf.log(self.prior_s2) - (self.prior_s1 + K) * tf.log(self.prior_s2 - tmp) - tmp

    @property
    @call_once
    def log_prior_pdf_per_dim(self):
        tmp = smooth_log(1 - self.vs)
        return tf.log(self.prior_s1) +  self.prior_s1 * tf.log(self.prior_s2) -\
            (self.prior_s1 + 1) * tf.log(self.prior_s2 - tmp) - tmp

    def sample_from_prior(self, num):
        betas = np.random.gamma(scale=self.prior_s1, shape=1./self.prior_s2, size=(num,))
        eps = np.floatX(np.stack([np.random.beta(self.prior_a, beta, size=(self.topic_dim-1,)) for beta in betas]))
        return self.stick_breaking(eps)

    def calc_kl_loss(self):
        a_mul_b = self.a * self.b
        Epv_a_qv_qa = - (tf.digamma(self.post_gamma1) - tf.log(self.post_gamma2))
        Epv_a_qv_qa += (self.post_gamma1 / self.post_gamma2 - 1) * tf.add_n([self.b / (m + a_mul_b) * _beta_fn(m / self.a, self.b) for m in range(1, 101)])

        log_entro = tf.log(a_mul_b)
        log_entro += (self.a - self.prior_a) / self.a * (-0.57721 - tf.digamma(self.b) - 1 / self.b)
        log_entro += -(self.b - 1) / self.b
        self.kl_v_3d = log_entro + Epv_a_qv_qa
        kl_v = tf.reduce_sum(self.kl_v_3d, axis=-1)

        kl_gamma = self.prior_s1 * (tf.log(self.post_gamma2) - tf.log(self.prior_s2)) - tf.lgamma(self.post_gamma1) + tf.lgamma(self.prior_s1) + (self.post_gamma1 - self.prior_s1) * tf.digamma(self.post_gamma1) - (self.post_gamma2 - self.prior_s2) * self.post_gamma1 / self.post_gamma2 # checked
        kl = kl_v + kl_gamma

        # This is 3-d variational lower bound of the K-L divergence KL(q(v) || p(v))
        # p(v) = \int p(v | b) p(b) \diff{b}
        # The prior pdf of the latent representation $v$ is p(v) = s_1 s_2 ** s_1 / (1-v) (s_2 - ln(1-z)) ** (s_1 + 1)
        # However E_{q(v)}[p(v)] is hard to calculate, so we use a variational lower bound
        # E_{q(v)q(b)}[ln(p(v,b)/q(b))] to approximate E_{q(v)}[p(v)].
        # As v_i is not independent under p(v) (conditional independent given b),
        # kl != tf.reduce_sum(self.KL_loss_3d, axis=-1)
        self.KL_loss_3d = self.kl_v_3d + kl_gamma

        return kl
    
    def calc_kl_loss_integral_out(self):
        log_entro = tf.log(a_mul_b)
        log_entro += (self.a - self.prior_a) / self.a * (-0.57721 - tf.digamma(self.b) - 1 / self.b)
        log_entro += -(self.b - 1) / self.b

        # As we cannot calculate the K-L divergence between the Kumar posterior and the integrated prior
        # we use monte-carlo sample from $q(v|x)$ to calculate $E_{q(v|x)}[\log p(v)]$ too,
        # which may lead to higher variance.
        kl = tf.reduce_sum(log_entro, axis=-1) - self.log_prior_pdf

        self.KL_loss_3d = log_entro - self.log_prior_pdf_per_dim

        return kl

    def build_kl_loss(self):
        # The prior.
        self.prior_a = np.floatX(self.cfg["prior_alpha"]) # Must be 1 here
        self.prior_s1 = np.floatX(self.cfg["prior_s1"])
        self.prior_s2 = np.floatX(self.cfg["prior_s2"])
        if self.cfg["integral_out"]:
            # When beta is integral out, this two parameter of the variational posterior is of no use
            # Let's set it to a constant.
            self.log_post_gamma1 = tf.constant(np.log(np.exp(self.cfg["prior_s1"]) - 1), dtype=tf.floatX, shape=[])
            self.log_post_gamma2 = tf.constant(np.log(np.exp(self.cfg["prior_s2"]) - 1), dtype=tf.floatX, shape=[])
            #self.log_post_gamma1 = tf.get_variable("sgd_log_post_gamma1", shape=[], dtype=tf.floatX, initializer=tf.constant_initializer(np.log(self.cfg["prior_s1"])))
            #self.log_post_gamma2 = tf.get_variable("sgd_log_post_gamma2", shape=[], dtype=tf.floatX, initializer=tf.constant_initializer(np.log(self.cfg["prior_s2"])))
            # self.post_gamma1 = tf.exp(self.log_post_gamma1)
            # self.post_gamma2 = tf.exp(self.log_post_gamma2)
        else:
            self.log_post_gamma1 = tf.get_variable("sgd_log_post_gamma1", shape=[], dtype=tf.floatX, initializer=tf.constant_initializer(np.log(np.exp(self.cfg["prior_s1"]) - 1)))
            self.log_post_gamma2 = tf.get_variable("sgd_log_post_gamma2", shape=[], dtype=tf.floatX, initializer=tf.constant_initializer(np.log(np.exp(self.cfg["prior_s2"]) - 1)))

        self.post_gamma1 = tf.nn.softplus(self.log_post_gamma1)
        self.post_gamma2 = tf.nn.softplus(self.log_post_gamma2)
        self.prior_b_mean = self.post_gamma1 / self.post_gamma2

        if self.cfg["integral_out"]:
            self.KL_Loss = self.calc_kl_loss_integral_out()
        else:
            self.KL_loss = self.calc_kl_loss()

        self.batch_kl_loss = tf.reduce_mean(self.KL_loss)

