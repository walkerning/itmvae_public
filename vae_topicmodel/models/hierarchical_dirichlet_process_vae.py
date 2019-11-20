# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import sys

import numpy as np
import tensorflow as tf
RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical

from .dirichlet_process_vae import DirichletProcessVAE
from .tf_backport import logdet, smooth_log
from vae_topicmodel.utils import call_once

def _cumsum(arr, axis=None, reverse=False, exclusive=False):
    if axis is None:
        axis = -1
        arr = arr.reshape(-1)
    if reverse:
        arr = np.flip(arr, axis=axis)
    res = np.cumsum(arr, axis=axis)
    if exclusive:
        res = np.concatenate([np.zeros_like(np.take(res, [0], axis=axis)),
                              np.take(res, range(0, arr.shape[axis]-1), axis=axis)],
                             axis=axis)
    if reverse:
        res = np.flip(res, axis=axis)
    return res

def _beta_fn(a, b):
    log_beta_fn = tf.lgamma(a) + tf.lgamma(b) - tf.lgamma(a + b)
    return tf.exp(log_beta_fn)

def get_dirichlet_ab_fct(name):
    mapping = {
        "exp": tf.exp,
        "softplus": tf.nn.softplus
    }
    return mapping[name]

def gumbel_softmax(probs, training_placeholder, tau_init=1.0, tau_trainable=False, MC_samples=1, straight_through=True):
    K = int(probs.get_shape()[-1])
    tau = tf.get_variable("gs_temperature", initializer=tau_init, trainable=tau_trainable)
    dist = RelaxedOneHotCategorical(tau, probs=probs)
    test_dist = tf.contrib.distributions.OneHotCategorical(probs=probs)
    test_samples = tf.cast(test_dist.sample(MC_samples), tf.floatX)
    y = dist.sample(MC_samples)
    if straight_through:
        y_hard = tf.cast(tf.one_hot(tf.argmax(y,-1),K), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    y = tf.where(training_placeholder, y, test_samples)
    return y, tau

class HierarchicalDirichletProcessVAE(DirichletProcessVAE):
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
        # "beta_regularizer": 0.001, # if `sage` is used
        # "sage_use_bias": True, # `sage`
        "beta_regularizer": None, # if `sage` is used
        "sage_use_bias": False, # `sage`
        "sage_bias_init": False,

        # regularizer configs
        "inference_regularizer": None,

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
        "pitman_yor": False,
        "prior_alpha": 1.,
        "prior_beta": 5.,

        # summary
        "summary_dir": None,

        # Stochastic layer
        "dirichlet_ab_fct": "softplus",
        "dirichlet_ab_use_bias": True,
        "dirichlet_phi_use_bias": True,
        "MC_samples": 1,

        "centering_input": False,

        # stick breaking
        "effective_indicator": "assignment", # one of `average`, `assignment`, `ratio`
        "stick_epsilon": 0.05, # 95% stick coverage
        "effective_threshold": 0.02, # average reaches 2%... not a very good definition...
        "assignment_threshold": 0.005, # at least 0.5% document is assigned to this topic
        "ratio_threshold": 0.1,
        "num_kl_term": 15,

        "decompose_beta_dim": None,
        "beta_hidden_topic_dim": [],
        "beta_hidden_word_dim": [],

        "b_init": 0,

        "encoder_log_input": False,
        "diversity_loss_type": None,

        # hierarhical
        "L2_truncation_level": 50,
        "gumbel_tau_init": 1.0,
        "gumbel_tau_trainable": False,
        "prior_gamma_a": 1.0, # only used when pitman_yor == True; otherwise set it to 1.0 please.
        "prior_gamma": 20.0,
        "KL_beta_ratio": 0.0,
        "posterior_one_c": False,

        # whether to train corpus-level beta posterior or update corpus-level beta posterior in closed form
        "closed_form_update_beta": False,
        "closed_form_update_sepoch": 0,
        "closed_form_update_every": 1,
        "closed_form_update_rate": False,
        "closed_form_prior_ratio": None,
        "closed_form_random": False,
        "closed_form_maxstep": None
    }

    _cfg_handlers = {
        "transfer_fct": lambda name: getattr(tf.nn, name),
        "dirichlet_ab_fct": get_dirichlet_ab_fct,
        "logvar_weight_init": lambda name: tf.contrib.layers.xavier_initializer() if name == "xavier" else tf.constant_initializer(name)
    }

    def __init__(self, cfg, train_cfg):
        super(HierarchicalDirichletProcessVAE, self).__init__(cfg, train_cfg)
        assert self.cfg["effective_indicator"] in {"average", "assignment", "ratio"}
        if self.cfg["closed_form_update_rate"]:
            assert "closed_form_update_rate" in self.train_cfg["schedule"], "`closed_form_update_rate` should be provided in train_cfg[\"schedule\"]"
            self.closed_form_update_rate = tf.placeholder(tf.float32, shape=[], name="closed_form_update_rate")

    @property
    @call_once
    def log_posterior_pdf(self):
        post_pi = tf.reduce_sum(tf.log(self.a) + tf.log(self.b) + (self.a - 1) * tf.log(self.vs) + (self.b - 1) * smooth_log(1 - self.vs ** self.a), axis=-1)
        post_c = tf.reduce_sum(smooth_log(tf.reduce_sum(self.c_4d * self.phi_prob, axis=-1)), axis=-1)
        return post_pi + post_c

    @property
    @call_once
    def log_prior_pdf(self):
        # For stick-breaking, prior_a = 1
        post_e_prob = self.stick_breaking_tensor(self.post_e_beta)
        c_prior_prob  = tf.reshape(post_e_prob, [1, 1, 1, self.topic_dim])
        prior_pi = tf.reduce_sum(tf.log(self.prior_b) + (self.prior_b - 1) * smooth_log(1 - self.vs), axis=-1)# + (self.prior_b - 1) * smooth_log(0.)
        prior_c = tf.reduce_sum(smooth_log(tf.reduce_sum(self.c_4d * c_prior_prob, axis=-1)), axis=-1)
        return prior_pi + prior_c

    # @property
    # @call_once
    # def log_beta_post_prior(self):
    #     post_beta_dist = tf.contrib.distributions.Beta(self.post_u, self.post_v)
    #     prior_beta_dist = tf.contrib.distributions.Beta(np.ones(self.topic_dim-1, dtype=np.floatX), self.cfg["prior_gamma"] * np.ones(self.topic_dim-1, dtype=np.floatX))
    #     post_beta = post_beta_dist.sample(self.cfg["MC_samples"])
    #     return tf.reduce_sum(smooth_log(post_beta_dist.prob(post_beta)), axis=-1), tf.reduce_sum(smooth_log(prior_beta_dist.prob(post_beta)), axis=-1), self.stick_breaking_tensor(post_beta)

    # @property
    # @call_once
    # def log_posterior_pdf(self):
    #     post_beta, _, _ = self.log_beta_post_prior
    #     post_pi = tf.reduce_sum(tf.log(self.a) + tf.log(self.b) + (self.a - 1) * tf.log(self.vs) + (self.b - 1) * smooth_log(1 - self.vs ** self.a), axis=-1)
    #     post_c = tf.reduce_sum(smooth_log(tf.reduce_sum(self.c_4d * self.phi_prob, axis=-1)), axis=-1)
    #     return tf.expand_dims(post_beta, -1) + post_pi + post_c

    # @property
    # @call_once
    # def log_prior_pdf(self):
    #     # For stick-breaking, prior_a = 1
    #     _, prior_beta, c_prior_prob = self.log_beta_post_prior # c_prior_prob is of (MC samples) * (topic_dim)
    #     c_prior_prob  = tf.reshape(c_prior_prob, [self.cfg["MC_samples"], 1, 1, self.topic_dim])
    #     prior_pi = tf.reduce_sum(tf.log(self.prior_b) + (self.prior_b - 1) * smooth_log(1 - self.vs), axis=-1)# + (self.prior_b - 1) * smooth_log(0.)
    #     prior_c = tf.reduce_sum(smooth_log(tf.reduce_sum(self.c_4d * c_prior_prob, axis=-1)), axis=-1)
    #     return tf.expand_dims(prior_beta, -1) + prior_pi + prior_c

    def stick_breaking_tensor(self, breaks):
        stick_segments_lst = []
        break_len = int(breaks.get_shape()[-1])
        reshape_breaks = tf.reshape(breaks, [-1, break_len])
        remaining_sticks = tf.ones([tf.shape(reshape_breaks)[0]], dtype=tf.floatX)
        for i in range(break_len):
            stick_segments_lst.append(remaining_sticks * reshape_breaks[:, i])
            remaining_sticks = remaining_sticks * (1 - reshape_breaks[:, i])
        stick_segments = tf.stack(stick_segments_lst) # (break_len) x (num)
        prob = tf.transpose(tf.concat((stick_segments, tf.expand_dims(remaining_sticks, axis=0)), axis=0), (1, 0))
        return tf.reshape(prob, tf.concat([tf.shape(breaks)[:-1], [break_len+1]], axis=0))

    def build_stochastic_layer(self, layer):
        self.a = tf.layers.dense(layer, self.cfg["L2_truncation_level"]-1, activation=self.cfg["dirichlet_ab_fct"],
                                 use_bias=self.cfg["dirichlet_ab_use_bias"],
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.zeros_initializer(),
                                 name="posterior_a_output")
        self.b = tf.layers.dense(layer, self.cfg["L2_truncation_level"]-1, activation=self.cfg["dirichlet_ab_fct"],
                                 use_bias=self.cfg["dirichlet_ab_use_bias"],
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.constant_initializer(self.cfg["b_init"]),
                                 name="posterior_b_output")
        uniform_samples = tf.random_uniform((self.cfg["MC_samples"], tf.shape(self.x)[0], self.cfg["L2_truncation_level"]-1), minval=0.01, maxval=0.99, dtype=tf.floatX)
        self.a = self.a + 1e-5
        self.b = self.b + 1e-5
        self.vs = (1 - uniform_samples ** (1 / self.b)) ** (1 / self.a)

        stick_segments_lst = []
        remaining_sticks = tf.ones((self.cfg["MC_samples"], tf.shape(self.x)[0]), dtype=tf.floatX)
        for i in range(self.cfg["L2_truncation_level"] - 1):
            stick_segments_lst.append(remaining_sticks * self.vs[:, :, i])
            remaining_sticks = remaining_sticks * (1 - self.vs[:, :, i])
        stick_segments = tf.stack(stick_segments_lst) # (self.cfg["L2_truncation_level"] - 1) x (MC samples) x (batch size)
        self.L2_z_3d = tf.transpose(tf.concat((stick_segments, tf.expand_dims(remaining_sticks, axis=0)), axis=0), (1, 2, 0))

        if not self.cfg["posterior_one_c"]:
            # multinomial logits
            self.phi_logits = tf.layers.dense(layer, self.cfg["L2_truncation_level"] * self.topic_dim, activation=None,
                                              use_bias=self.cfg["dirichlet_phi_use_bias"],
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              bias_initializer=tf.zeros_initializer(),
                                              name="posterior_phi_output")
            self.phi_prob = tf.nn.softmax(tf.reshape(self.phi_logits,
                                                     [-1, self.cfg["L2_truncation_level"], self.topic_dim]))
        else:
            self.phi_logits = tf.layers.dense(layer, self.topic_dim, activation=None,
                                              use_bias=self.cfg["dirichlet_phi_use_bias"],
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              bias_initializer=tf.zeros_initializer(),
                                              name="posterior_phi_output")
            self.phi_prob = tf.tile(tf.expand_dims(tf.nn.softmax(tf.reshape(self.phi_logits,
                                                                            [-1, self.topic_dim])),
                                                   axis=1), [1, self.cfg["L2_truncation_level"], 1])
        # c_4d: (MC samples) x (batch size) x (L2 truncation level) x (L1 truncation level/topic dim)
        self.soft_z_3d = tf.reduce_sum(tf.expand_dims(self.L2_z_3d, axis=-1) * self.phi_prob, axis=2)
        self.c_4d, self.gumbel_tau = gumbel_softmax(self.phi_prob, self.training_placeholder,
                                                    tau_init=self.cfg["gumbel_tau_init"], tau_trainable=self.cfg["gumbel_tau_trainable"],
                                                    MC_samples=self.cfg["MC_samples"], straight_through=True)
        self.z_3d = tf.reduce_sum(tf.expand_dims(self.L2_z_3d, axis=-1) * self.c_4d, axis=2)
        
        # Change
        if self.cfg["effective_indicator"] == "average":
            self.average_of_every_topic = tf.reduce_mean(self.z_3d, axis=(0, 1)) * tf.cast(tf.shape(self.x)[0], tf.floatX)
            effective_dims = self.average_of_every_topic > self.cfg["effective_threshold"]
            self.average_used_dims = tf.reduce_sum(tf.cast(effective_dims, tf.floatX))
            self.effective_dims = tf.squeeze(tf.where(effective_dims))
        elif self.cfg["effective_indicator"] == "assignment" or self.cfg["effective_indicator"] == "ratio":
            self.assignment_of_every_topic = tf.bincount(tf.cast(tf.argmax(self.z_3d, axis=-1), tf.int32), minlength=self.topic_dim)
            effective_dims_bool = tf.cast(self.assignment_of_every_topic, tf.floatX) > self.cfg["assignment_threshold"] * tf.cast(tf.shape(self.x)[0], tf.floatX) * self.cfg["MC_samples"]
            # FIXME: for now, if MC_sample is not 1. This is not correct.
            self.average_used_dims = tf.reduce_sum(tf.cast(effective_dims_bool, tf.floatX))
            self.effective_dims = tf.squeeze(tf.where(effective_dims_bool))

        # self.average_used_dims = tf.Print(self.average_used_dims, [tf.transpose(remaining_sticks, (1, 2, 0))], "print_remaining", summarize=100, first_n=3)
        # self.z = tf.Print(self.z, [self.z], "print_z", summarize=50)
        # self.z = tf.Print(self.z, [tf.reduce_sum(self.z, axis=-1)], "print_z_sum")
        z = tf.reshape(self.z_3d, [-1, self.topic_dim])
        return z

    def calc_kl_loss(self):
        # KL(q(pi)||p(pi))
        a_mul_b = self.a * self.b
        # kl = 1. / (1 + a_mul_b) * _beta_fn(1./self.a, self.b)
        kl = tf.add_n([1. / (m + a_mul_b) * _beta_fn(m/self.a, self.b) for m in range(1, self.cfg["num_kl_term"]+1)])
        # kl = tf.Print(kl, [self.a, self.b], summarize=100)
        kl = tf.check_numerics(kl, "check_kl2", "check_kl2")
        kl = kl * (self.prior_b - 1) * self.b
        kl = tf.check_numerics(kl, "check_kl3", "check_kl3")
        # psi_b_taylor_approx = tf.log(self.b) - 1./(2 * self.b) - 1./(12 * self.b**2)
        kl += (self.a - self.prior_a) / self.a * (-0.57721 - tf.digamma(self.b) - 1 / self.b)
        kl = tf.check_numerics(kl, "check_kl4", "check_kl4")
        kl += tf.log(a_mul_b) + tf.log(_beta_fn(self.prior_a, self.prior_b))
        kl = tf.check_numerics(kl, "check_kl5", "check_kl5")
        kl += -(self.b - 1) / self.b
        # kl = tf.Print(kl, [tf.shape(kl)], 'print_kl_shape')
        # return tf.reduce_sum(kl, axis=-1)
        return kl

    def calc_kl_loss_beta(self):
        u_plus_v = self.post_u + self.post_v
        if self.cfg.get("pitman_yor", False):
            c = tf.reduce_sum(- tf.lgamma(self.prior_gamma + self.prior_gamma_a) + tf.lgamma(self.prior_gamma) + tf.lgamma(self.prior_gamma_a))
            return c - tf.reduce_sum(tf.lgamma(self.post_u) + tf.lgamma(self.post_v) - tf.lgamma(u_plus_v) + (tf.digamma(u_plus_v) - tf.digamma(self.post_u)) * (self.post_u - self.prior_gamma_a) + (tf.digamma(u_plus_v) - tf.digamma(self.post_v)) * (self.post_v - self.prior_gamma))
        else:
            return -(self.topic_dim - 1) * np.log(self.prior_gamma) - tf.reduce_sum(tf.lgamma(self.post_u) + tf.lgamma(self.post_v) - tf.lgamma(u_plus_v) + (tf.digamma(u_plus_v) - tf.digamma(self.post_u)) * (self.post_u - self.prior_gamma_a) + (tf.digamma(u_plus_v) - tf.digamma(self.post_v)) * (self.post_v - self.prior_gamma))

    def calc_kl_loss_c(self):
        # entropy = -tf.reduce_sum(self.phi_prob * tf.log(self.phi_prob), axis=-1)
        # #entropy = tf.Print(entropy, [-entropy], "minus_entropy")
        # # Aji = tf.reduce_sum(self.phi_prob, axis=1) # (batch size) x (topic dim/L1 truncation level)
        # Aji = tf.reduce_sum(self.phi_prob, axis=1)[:, :self.topic_dim-1]
        # sAji = tf.cumsum(Aji, axis=-1, reverse=True, exclusive=True)
        # E_psilogbeta = tf.reduce_sum(Aji * tf.digamma(self.post_u) - (sAji+Aji) * tf.digamma(self.post_u + self.post_v) + sAji * tf.digamma(self.post_v), axis=-1)

        entropy = -tf.reduce_sum(self.phi_prob * tf.log(self.phi_prob + 1e-8), axis=-1)
        Aji = tf.reduce_sum(self.phi_prob, axis=1) # (batch size) x (topic dim/L1 truncation level)
        Aji_nolast = Aji[:, :self.topic_dim-1] # without the last dimension(L1 truncation level)
        sAji = tf.cumsum(Aji, axis=-1, reverse=True, exclusive=True)[:, :self.topic_dim-1]
        E_psilogbeta = tf.reduce_sum(Aji_nolast * tf.digamma(self.post_u) - (sAji+Aji_nolast) * tf.digamma(self.post_u + self.post_v) + sAji * tf.digamma(self.post_v), axis=-1)

        return -tf.reduce_sum(entropy, axis=-1) - E_psilogbeta

    def build_kl_loss(self):
        # The prior.
        self.prior_a = np.floatX(self.cfg["prior_alpha"])
        self.prior_b = np.floatX(self.cfg["prior_beta"])
        self.prior_gamma_a = np.floatX(self.cfg["prior_gamma_a"])
        self.prior_gamma = np.floatX(self.cfg["prior_gamma"])
        if self.cfg.get("pitman_yor", False):
            self.prior_b = np.floatX(np.arange(self.cfg["L2_truncation_level"] - 1) * (1 - self.prior_a) + self.prior_b)
            self.prior_gamma = np.floatX(np.arange(self.topic_dim - 1) * (1 - self.prior_gamma_a) + self.prior_gamma)
        if self.cfg["closed_form_update_beta"]:
            self.post_u = tf.get_variable("post_u", shape=[self.topic_dim-1], dtype=tf.floatX,
                                          initializer=tf.constant_initializer(1.0), trainable=False)
            self.post_v = tf.get_variable("post_v", shape=[self.topic_dim-1], dtype=tf.floatX,
                                          initializer=tf.constant_initializer(self.prior_gamma), trainable=False)
        else:
            self.inv_post_u = tf.get_variable("sgd_log_post_u", shape=[self.topic_dim-1], dtype=tf.floatX,
                                              initializer=tf.constant_initializer(np.log(np.exp(1.0)-1)))
            self.inv_post_v = tf.get_variable("sgd_log_post_v", shape=[self.topic_dim-1], dtype=tf.floatX,
                                              initializer=tf.constant_initializer(np.log(np.exp(self.prior_gamma)-1)))
            self.post_u = tf.nn.softplus(self.inv_post_u)
            self.post_v = tf.nn.softplus(self.inv_post_v)
        self.post_e_beta = self.post_u / (self.post_u + self.post_v)

        self.KL_pi = self.calc_kl_loss()
        self.KL_beta = self.calc_kl_loss_beta()
        self.KL_c = self.calc_kl_loss_c()
        # self.KL_loss_3d = self.calc_kl_loss()
        # self.KL_loss = tf.reduce_sum(self.KL_loss_3d, axis=-1)
        self.batch_kl_c = tf.reduce_mean(self.KL_c)
        self.batch_kl_pi = tf.reduce_mean(tf.reduce_sum(self.KL_pi, axis=-1))
        self.KL_loss = tf.reduce_sum(self.KL_pi, axis=-1) + self.KL_c + self.KL_beta / self.dataset_size_placeholder
        if self.cfg["KL_beta_ratio"] > 0:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.cfg["KL_beta_ratio"] * self.KL_beta)
        # / tf.where(self.training_placeholder,

        # / tf.cast(tf.shape(self.x)[0] * self.cfg["MC_samples"], tf.float32)
        # self.KL_loss = tf.Print(self.KL_loss, [tf.reduce_mean(tf.reduce_sum(self.KL_pi, axis=-1)), tf.reduce_mean(self.KL_c), self.KL_beta], "kl losses")

        self.diversity_loss = tf.constant(0., dtype=tf.floatX)

        self.batch_kl_loss = tf.reduce_mean(self.KL_loss)

    @property
    @call_once
    def _tmp_sum_phi_prob(self):
        return tf.reduce_sum(tf.reduce_sum(self.phi_prob, axis=0, keep_dims=True), axis=1)

    @property
    @call_once
    def _tmp_mean_phi_prob(self):
        return tf.reduce_sum(tf.reduce_mean(self.phi_prob, axis=0), axis=0)

    def corpus_calculate_u_v(self):
        # tmp = self.test_on_dataset("train", tensor=self._tmp_mean_phi_prob, use_sum=True)[:-1]
        maxstep = self.cfg["closed_form_maxstep"]
        tmp = self.test_on_dataset("train", tensor=self._tmp_mean_phi_prob, use_sum=True, random=self.cfg["closed_form_random"], maxstep=maxstep)
        step_ratio = 1.0
        if maxstep:
            step_ratio = self.reader.train_size / (maxstep * self.train_cfg["test_batch_size"])
        if self.cfg["closed_form_prior_ratio"] is not None:
            new_u = self.prior_gamma_a * self.cfg["closed_form_prior_ratio"] * self.reader.train_size + tmp[:-1] * step_ratio
            new_v = self.prior_gamma * self.cfg["closed_form_prior_ratio"] * self.reader.train_size + _cumsum(tmp, reverse=True, exclusive=True)[:-1] * step_ratio
        else:
            new_u = self.prior_gamma_a + tmp[:-1] * step_ratio
            new_v = self.prior_gamma + _cumsum(tmp, reverse=True, exclusive=True)[:-1] * step_ratio
        return new_u, new_v

    def on_epoch_end(self, epoch, schedule_dict):
        if self.cfg["closed_form_update_beta"]:
            if epoch >= self.cfg["closed_form_update_sepoch"] and epoch % self.cfg["closed_form_update_every"] == 0:
                print("Updating corpus-level beta...")
                # train_phi_prob = self.topic_prop_on_dataset("train", tensor=self._tmp_sum_phi_prob, axis=0)[:, :-1]
                new_u, new_v = self.corpus_calculate_u_v()
                if self.cfg["closed_form_update_rate"]:
                    now_u = self.sess.run(self.post_u)
                    now_v = self.sess.run(self.post_v)
                    # NOTE: closed_form_update_rate schedule serie should satisfy: the serie diverge and its square serie converge; See Wang 2011
                    new_u = now_u + schedule_dict[self.closed_form_update_rate] * (new_u - now_u)
                    new_v = now_v + schedule_dict[self.closed_form_update_rate] * (new_v - now_v)
                self.sess.run(tf.assign(self.post_u, new_u))
                self.sess.run(tf.assign(self.post_v, new_v))
        # **TODO**: batch natural gradient update?
