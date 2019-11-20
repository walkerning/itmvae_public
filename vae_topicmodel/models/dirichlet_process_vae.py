# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import sys

import numpy as np
import tensorflow as tf

from .vae import VAE
from .tf_backport import logdet, smooth_log
from vae_topicmodel.utils import call_once

def _beta_fn(a, b):
    log_beta_fn = tf.lgamma(a) + tf.lgamma(b) - tf.lgamma(a + b)
    return tf.exp(log_beta_fn)

def get_dirichlet_ab_fct(name):
    mapping = {
        "exp": tf.exp,
        "softplus": tf.nn.softplus
    }
    return mapping[name]

class DirichletProcessVAE(VAE):
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
        "prior_beta": 5., # > 0

        # summary
        "summary_dir": None,

        # Stochastic layer
        "dirichlet_ab_fct": "softplus",
        "dirichlet_ab_use_bias": True,
        "MC_samples": 1,
        "bias_on_prior": False,

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
        "diversity_loss_type": None
    }
    _cfg_handlers = {
        "transfer_fct": lambda name: getattr(tf.nn, name),
        "dirichlet_ab_fct": get_dirichlet_ab_fct,
        "logvar_weight_init": lambda name: tf.contrib.layers.xavier_initializer() if name == "xavier" else tf.constant_initializer(name)
    }

    def __init__(self, cfg, train_cfg):
        super(DirichletProcessVAE, self).__init__(cfg, train_cfg)
        assert self.cfg["effective_indicator"] in {"average", "assignment", "ratio"}

    @property
    @call_once
    def log_posterior_pdf(self):
        return tf.reduce_sum(tf.log(self.a) + tf.log(self.b) + (self.a - 1) * tf.log(self.vs) + (self.b - 1) * smooth_log(1 - self.vs ** self.a), axis=-1)

    @property
    @call_once
    def log_prior_pdf(self):
        # For stick-breaking, prior_a = 1
        # FIXME: pitman_yor not supported now !
        # assert not self.cfg.get("pitman_yor", False)
        return tf.reduce_sum(tf.log(self.prior_b) + (self.prior_b - 1) * smooth_log(1 - self.vs), axis=-1)# + (self.prior_b - 1) * smooth_log(0.)

    def stick_breaking(self, breaks):
        # Stick breaking process
        num = breaks.shape[0]
        remaining = np.ones((num,))
        sticks = np.zeros((num, self.topic_dim))
        for i in range(self.topic_dim - 1):
            sticks[:, i] = remaining * breaks[:, i]
            remaining = remaining * (1 - breaks[:, i])
        sticks[:, self.topic_dim-1] = remaining
        return sticks

    def sample_from_prior(self, num):
        eps = np.floatX(np.random.beta(self.prior_a, self.prior_b, size=(num, self.topic_dim-1)))
        return self.stick_breaking(eps)

    def build_stochastic_layer(self, layer):
        self.a = tf.layers.dense(layer, self.topic_dim-1, activation=self.cfg["dirichlet_ab_fct"],
                                 use_bias=self.cfg["dirichlet_ab_use_bias"],
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.zeros_initializer(),
                                 name="posterior_a_output")
        self.b = tf.layers.dense(layer, self.topic_dim-1, activation=self.cfg["dirichlet_ab_fct"],
                                 use_bias=self.cfg["dirichlet_ab_use_bias"],
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.constant_initializer(self.cfg["b_init"]),
                                 name="posterior_b_output")
        uniform_samples = tf.random_uniform((self.cfg["MC_samples"], tf.shape(self.x)[0], self.topic_dim-1), minval=0.01, maxval=0.99, dtype=tf.floatX)
        if self.cfg.get("bias_on_prior", False):
            self.prior_a = np.floatX(self.cfg["prior_alpha"])
            self.prior_b = np.floatX(self.cfg["prior_beta"])
            if self.cfg["pitman_yor"]:
                self.prior_b = np.floatX(np.arange(self.topic_dim - 1) * (1 - self.prior_a) + self.prior_b)
            self.b = self.b + self.prior_b
            self.a = self.a + self.prior_a
        else:
            self.a = self.a + 1e-5
            self.b = self.b + 1e-5
        self.vs = (1 - uniform_samples ** (1 / self.b)) ** (1 / self.a)

        # self.vs = tf.Print(self.vs, [tf.reduce_mean(self.vs), tf.reduce_max(self.vs), self.vs[:, 37, :]], summarize=200, message="print_vs: ")
        # Construct topic vector by stick-breaking process
        # stick_segment = tf.zeros((self.cfg["MC_samples"], tf.shape(self.x)[0]))
        # remaining_stick = tf.ones((self.cfg["MC_samples"], tf.shape(self.x)[0]))
        # def stick_breaking(s, elem):
        #     stick = s[1] * self.vs[:, :, elem]
        #     remain = s[1] * (1 - self.vs[:, :, elem])
        #     return (stick, remain)
        # stick_segments, remaining_sticks = tf.scan(fn=stick_breaking, elems=tf.range(self.topic_dim - 1),
        #                                            initializer=(stick_segment, remaining_stick))
        # self.z = tf.transpose(tf.concat((stick_segments, tf.expand_dims(remaining_sticks[-1, :, :], axis=0)), axis=0), (1, 2, 0))
        # # 0.01 -> 99% stick
        # self.average_used_dims = tf.reduce_mean(tf.reduce_sum(tf.cast(remaining_sticks > self.cfg["stick_epsilon"], tf.floatX), axis=0))

        stick_segments_lst = []
        remaining_sticks = tf.ones((self.cfg["MC_samples"], tf.shape(self.x)[0]), dtype=tf.floatX)
        for i in range(self.topic_dim - 1):
            stick_segments_lst.append(remaining_sticks * self.vs[:, :, i])
            remaining_sticks = remaining_sticks * (1 - self.vs[:, :, i])
        stick_segments = tf.stack(stick_segments_lst) # (topic_dim - 1) x (MC samples) x (batch size)
        self.z_3d = tf.transpose(tf.concat((stick_segments, tf.expand_dims(remaining_sticks, axis=0)), axis=0), (1, 2, 0))
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

    def set_threshold(self, threshold):
        if self.cfg["effective_indicator"] == "assignment":
            self.cfg["assignment_threshold"] = threshold
        elif self.cfg["effective_indicator"] == "average":
            self.cfg["effective_threshold"] = threshold
        elif self.cfg["effective_indicator"] == "ratio":
            self.cfg["ratio_threshold"] = threshold
        else:
            raise Exception("Invalid effective indicator.")

    @property
    def topic_components(self):
        # _, effective_dims = self._topic_components(data_type="train")
        _, effective_dims = self._topic_components(data_type="test")
        return self.sess.run(self._topic_components_tensor)[effective_dims]

    def _topic_components(self, data_type="train"):
        print("Use effective indicator: ", self.cfg["effective_indicator"])
        # Very ugly... use train_cfg in this class... in fact, log-input is not used now!
        if self.train_cfg["log_input"]:
            def _log_input(x):
                return np.log(1 + x)
        _feed_dct = {self.training_placeholder: False}
        # Find the average used topic number on the test dataset
        if self.train_cfg["test_batch_size"] is None:
            def _gen_wrap():
                yield self.reader.get_parsed_data_from_type(data_type, func=(_log_input if self.train_cfg["log_input"] else lambda x:x))
            test_data_gen = _gen_wrap()
        else:
            test_data_gen = self.reader.iterator_one_pass(self.train_cfg["test_batch_size"], data_type=data_type, func=(_log_input if self.train_cfg["log_input"] else lambda x:x))
        if self.cfg["effective_indicator"] == "ratio" or self.cfg["effective_indicator"] == "assignment":
            name = "assignment_of_every_topic"
        else:
            name = "average_of_every_topic"
        if self.cfg["effective_indicator"] == "average":
            eff_threshold = self.cfg["effective_threshold"]
        elif self.cfg["effective_indicator"] == "assignment":
            eff_threshold = self.cfg["assignment_threshold"]
        else:
            eff_threshold = self.cfg["ratio_threshold"]

        tensor = getattr(self, name)
        res = np.zeros(self.topic_dim, dtype=np.floatX)
        num_test = 0
        while 1:
            try:
                datum = test_data_gen.next()
            except StopIteration:
                break
            res += self._run(tensor, datum, _feed_dct)
            print("\rTesting ... {:5}".format(num_test), end="", file=sys.stderr)
            num_test += len(datum)
        res /= num_test * self.cfg["MC_samples"]
        print("\r", end="", file=sys.stderr)
        print("Finish test {:5} samples".format(num_test))
        if self.cfg["effective_indicator"] == "ratio":
            eff_threshold = np.max(res) * eff_threshold
        effective_dims = np.where(res > eff_threshold)[0]
        avg_used_dims = len(effective_dims)
        s_assign_max = np.max(res[effective_dims])
        s_assign_min = np.min(res[effective_dims])
        print("[indicator: {} threshold: {}] Average used dims: {}. {}. max/min ratio in selected: {} {}\n\t{}".format(self.cfg["effective_indicator"], eff_threshold, avg_used_dims, effective_dims, s_assign_max, s_assign_min, res))
        return res, effective_dims

    def calc_kl_loss(self):
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

    def build_kl_loss(self):
        # The prior.
        self.prior_a = np.floatX(self.cfg["prior_alpha"])
        self.prior_b = np.floatX(self.cfg["prior_beta"])
        if self.cfg["pitman_yor"]:
            # Pitman-Yor process for power-law cluster size distribution. 
            # self.prior_a is `1 - a` in the standard parametrization ( a is the discount parameter ) ; 0 < prior_a <= 1
            # self.prior_b is `a + b` in the standard parametrization; 0 < prior_b
            # As the cluster distribution goes with k^{-1/a} asymptotically; the larger the prior_a, the smaller the a, the less activated topics the prior encourage
            self.prior_b = np.floatX(np.arange(self.topic_dim - 1) * (1 - self.prior_a) + self.prior_b)
        self.KL_loss_3d = self.calc_kl_loss()
        self.KL_loss = tf.reduce_sum(self.KL_loss_3d, axis=-1)

        # diviserity loss
        dl_type = self.cfg.get("diversity_loss_type", None)
        print("Use diversity regularization: ", dl_type)
        if dl_type == "xie_2015":
            K = []
            beta_norm = tf.sqrt(tf.reduce_sum(tf.square(self.beta), axis=-1))
            for i in range(self.topic_dim):
                Ki = []
                for j in range(i):
                    Ki.append(K[j][i])
                Ki.append(tf.constant(0, dtype=tf.floatX))
                for j in range(i+1, self.topic_dim):
                    Ki.append(tf.acos(tf.reduce_sum(self.beta[i, :] * self.beta[j, :]) / (beta_norm[i] * beta_norm[j])))
    
                K.append(tf.stack(Ki))
            K_mat = tf.stack(K)
            self.angle_mean = tf.reduce_mean(K_mat)
            self.angle_v = tf.reduce_mean(tf.square(K_mat - self.angle_mean))
            self.diversity_loss = -self.diversity_weight_placeholder * (self.angle_mean - self.angle_v)
        elif dl_type == "dpp":
            K = []
            #beta_norm = tf.sqrt(tf.reduce_sum(tf.square(self.beta), axis=-1))
            for i in range(self.topic_dim):
                Ki = []
                for j in range(i):
                    Ki.append(K[j][i])
                #Ki.append(tf.constant(, dtype=tf.floatX))
                for j in range(i, self.topic_dim):
                    # TODO: other kernels?
                    Ki.append(tf.reduce_sum(self.beta[i, :] * self.beta[j, :]))
                K.append(tf.stack(Ki))
            K_mat = tf.stack(K)
            self.diversity_loss = - 2 * logdet(K_mat)

        else:
            self.diversity_loss = tf.constant(0., dtype=tf.floatX)

        self.batch_kl_loss = tf.reduce_mean(self.KL_loss)
            
