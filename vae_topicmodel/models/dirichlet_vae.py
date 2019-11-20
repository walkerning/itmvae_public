# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import copy

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

from .gaussian_vae import GaussianVAE

def get_dirichlet_a_fct(name):
    mapping = {
        "exp": tf.exp,
        "softplus": tf.nn.softplus
    }
    return mapping[name]
        
class DirichletVAE(GaussianVAE):
    MODEL_NAME = "DirichletVAE"
    _default_cfg = {
        # network structure
        "inference_net_structure": [100, 100],
        "gen_net_structure": [],
        "transfer_fct": "softplus",
        "use_residual": False,
        "softmax_topic_vector": False,
        "dropout_topic_vector": False,
        "topic_role": "mixture", # could be one of `mixture`, `sage`
        "beta_regularizer": 0.001, # if `sage` is used
        "sage_use_bias": True, # `sage`

        # NVIL
        "baseline_net_structure": [100],
        "baseline_transfer_fct": "tanh",
        "variance_normalization": True,
        "nvil_decay": 0.8,

        # batch norm configs
        "batch_norm_gen_net": False,
        "batch_norm_inference_net": False,
        "batch_norm_baseline_net": False,
        "batch_norm_logits_wdist": True, # `sage`
        "batch_norm_beta": True, # `mixture`

        # initialize configs
        "logvar_bias_init": -4,
        "logvar_weight_init": 0,
        
        # prior configs
        "prior_alpha": None,

        # summary
        "summary_dir": None,

        # Stochastic layer
        "dirichlet_a_fct": "softplus",
        "MC_samples": 1
    }
    _cfg_handlers = {
        "transfer_fct": lambda name: getattr(tf.nn, name),
        "baseline_transfer_fct": lambda name: getattr(tf.nn, name),
        "dirichlet_a_fct": get_dirichlet_a_fct,
        "logvar_weight_init": lambda name: tf.contrib.layers.xavier_initializer() if name == "xavier" else tf.constant_initializer(name)
    }

    def __init__(self, cfg, train_cfg):
        super(DirichletVAE, self).__init__(cfg, train_cfg)

    def init(self, test_only=False):
        if not test_only:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.optimizer_gen = getattr(tf.train, self.train_cfg["optimizer"])(**self.train_cfg["optimizer_cfg"])
            inference_cfg = copy.deepcopy(self.train_cfg["optimizer_cfg"])
            # use learning rate that is 5 times smaller than for the model
            inference_cfg["learning_rate"] = inference_cfg["learning_rate"] / 5
            self.optimizer_inference = getattr(tf.train, self.train_cfg["optimizer"])(**inference_cfg)
            with tf.control_dependencies(update_ops):
                self.opt_step_gen = self.optimizer_gen.minimize(self.gen_surrogate)
                self.opt_step_inference = self.optimizer_inference.minimize(self.inference_surrogate + self.KL_weight_placeholder * self.batch_kl_loss + self.baseline_surrogate)
                self.opt_step = tf.group(self.opt_step_gen, self.opt_step_inference)

        self.sess.run(tf.global_variables_initializer())

        if not test_only and self.cfg["summary_dir"]:
            self.summary_writer = tf.summary.FileWriter(self.cfg["summary_dir"], self.sess.graph)
            self.summary_merged = tf.summary.merge_all()
        else:
            self.summary_writer = None

    def build_baseline_net(self, x):
        bnet_structure = self.cfg["baseline_net_structure"]
        layer = x
        for i, dim in enumerate(bnet_structure):
            layer = tf.layers.dense(layer, dim, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                    name="baseline_layer_{}".format(i+1))
            if self.cfg["batch_norm_baseline_net"]:
                layer = tf.contrib.layers.batch_norm(layer, is_training=self.training_placeholder)
            layer = self.cfg["baseline_transfer_fct"](layer)
        baseline = tf.layers.dense(layer, 1, activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                    name="baseline_output")
        return baseline

    def build_stochastic_layer(self, layer):
        self.posterior_a = tf.layers.dense(layer, self.topic_dim, activation=self.cfg["dirichlet_a_fct"],
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           bias_initializer=tf.zeros_initializer(),
                                           name="posterior_a_output") + 1e-4
        self.posterior_a = tf.check_numerics(self.posterior_a, "check_self_posterior_a")
        # self.posterior_a = tf.Print(self.posterior_a, [self.posterior_a], "print_self_posterior_a")
        self.posterior_a0 = tf.reduce_sum(self.posterior_a, axis=-1, keep_dims=True)
        self.dist = tf.contrib.distributions.Dirichlet(self.posterior_a)
        # sample operation do not add add gradient op automatically. no need to stop gradient
        self.z = self.dist.sample(self.cfg["MC_samples"])
        self.z = tf.check_numerics(self.z, "check_self_z")
        # self.z = tf.Print(self.z, [self.z], "self_z", name="print_self_z")
        z = tf.reshape(self.z, [-1, self.topic_dim])
        return z

    def build_loss(self):
        # The prior.
        self.prior_a = self.cfg["prior_alpha"] * np.ones([self.topic_dim], dtype=np.float32)
        self.prior_a0 = np.float32(self.cfg["prior_alpha"] * self.topic_dim)

        # baseline net
        self.baseline = tf.squeeze(self.build_baseline_net(self.x))
        # self.baseline = 0

        # Let's smooth the multinomial parameters.
        self.w_dist += 1e-10
        # reshape back to size `MC_sample x batch_size`
        self.w_dist_3d = tf.reshape(self.w_dist, [self.cfg["MC_samples"], -1, self.cfg["vocab_dim"]])
        
        # For generation parameters
        log_p_x_z = tf.reduce_sum(self.x * tf.log(self.w_dist_3d), axis=-1) # log P(x | z)
        log_p_x_z = tf.check_numerics(log_p_x_z, "check_log_p_x_z") # 应该要是log P(x|z) + log P(z). 但是P(z)先验没有参数...
        self.gen_surrogate = - tf.reduce_mean(log_p_x_z)

        # For baseline network parameters
        # q = self.dist.log_prob(self.z) # log Q(z | x)
        from tensorflow.python.ops import special_math_ops
        from tensorflow.python.ops import math_ops
        logz = math_ops.log(self.z + 1e-4)
        logz = tf.check_numerics(logz, "check_logz")
        unnorm = math_ops.reduce_sum((self.posterior_a - 1.) * logz, -1)
        unnorm = tf.check_numerics(unnorm, "check_unnorm")
        # unnorm = tf.Print(tf.check_numerics(unnorm, "check_unnorm"), [tf.shape(unnorm)], "print_unorm_shape")
        # normfact = special_math_ops.lbeta(self.posterior_a)
        # normfact = tf.check_numerics(normfact, "check_normfact")
        # 拆normfact
        lgamma_posterior_a = math_ops.lgamma(self.posterior_a)
        lgamma_posterior_a = tf.check_numerics(lgamma_posterior_a, "check_lgamma_posterior_a")
        log_prod_gamma_x = math_ops.reduce_sum(lgamma_posterior_a, reduction_indices=[-1])
        log_prod_gamma_x = tf.check_numerics(log_prod_gamma_x, "check_log_prod_gamma_x")
        log_gamma_sum_x = math_ops.lgamma(tf.squeeze(self.posterior_a0))
        normfact = log_prod_gamma_x - log_gamma_sum_x
        # normfact = tf.Print(tf.check_numerics(normfact, "check_normfact"), [tf.shape(normfact)], "print_normfact_shape")
        q = unnorm - normfact
        
        q = tf.check_numerics(q, "check_q")
        l_signal = log_p_x_z # log P(x | z)
        # - q # log P(x | z) - log Q(z | x)
        l_signal = tf.check_numerics(l_signal, "check_l_signal")
        l_signal_mean, l_signal_variance = tf.nn.moments(l_signal, axes=[1])
        tf.summary.histogram("batch mean of l sginal", l_signal_mean)
        tf.summary.histogram("batch variance of l sginal", l_signal_variance)
        # baseline is of size `batch_size`
        self.baseline_surrogate = tf.reduce_mean(0.5 * tf.square(tf.stop_gradient(l_signal) - self.baseline))

        # For inference network parameter
        l_signal = l_signal - self.baseline
        # Variance normalization
        if self.cfg["variance_normalization"]:
            nvil_decay = self.cfg["nvil_decay"]
            bc = tf.reduce_mean(l_signal)
            bv = tf.reduce_mean(tf.square(l_signal - bc))
            moving_mean = tf.get_variable(
                'moving_mean', shape=[], initializer=tf.constant_initializer(0.),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', shape=[],
                initializer=tf.constant_initializer(1.), trainable=False)
    
            update_mean = moving_averages.assign_moving_average(
                moving_mean, bc, decay=nvil_decay)
            update_variance = moving_averages.assign_moving_average(
                moving_variance, bv, decay=nvil_decay)
            l_signal = (l_signal - moving_mean) / tf.maximum(
                1., tf.sqrt(moving_variance))
            with tf.control_dependencies([update_mean, update_variance]):
                l_signal = tf.identity(l_signal)

        l_signal_mean_after, l_signal_variance_after = tf.nn.moments(l_signal, axes=[1])
        tf.summary.histogram("batch mean of l sginal after vairance normalization", l_signal_mean_after)
        tf.summary.histogram("batch variance of l sginal after variance normalization", l_signal_variance_after)

        self.inference_surrogate = tf.reduce_mean(- tf.stop_gradient(l_signal) * q)
        # gradient of minimizing this surrogate term will be `l_signal * \grad_\phi log Q(z|x)`

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = tf.add_n(reg_losses) if reg_losses else 0.

        # K-L divergence of two Dirichlet is writen out analytically
        # - A(posterior_a) + <(posterior_a - prior_a), \grad_a A(posterior_a)> + A(prior_a)
        self.kl_loss = - (tf.reduce_sum(tf.lgamma(self.posterior_a), axis=-1) - tf.lgamma(self.posterior_a0)) + \
                  tf.reduce_sum((self.posterior_a - self.prior_a) * (tf.digamma(self.posterior_a) - tf.digamma(self.posterior_a0)), axis=-1) + tf.reduce_sum(tf.lgamma(self.prior_a), axis=-1) - tf.lgamma(self.prior_a0)
        self.kl_loss = tf.check_numerics(self.kl_loss, "check_kl_loss")
        self.batch_rec_loss = self.gen_surrogate
        self.batch_kl_loss = tf.reduce_mean(self.kl_loss)
        self.loss = self.batch_rec_loss + self.batch_kl_loss

        # In batch mode, log perplexity of an epoch might not be the same
        # as the log perplexity of the whole dataset...
        self.log_perplexity = self.loss / tf.reduce_sum(self.x) * tf.cast(tf.shape(self.x)[0], tf.float32)

        # self.opt_surrogate = self.gen_surrogate + self.baseline_surrogate + self.inference_surrogate + self.reg_loss
        self.opt_surrogate = self.gen_surrogate + self.baseline_surrogate + self.batch_kl_loss + self.inference_surrogate + self.reg_loss

        tf.summary.scalar("batch reconstruction loss(gen surrogate)", self.batch_rec_loss)
        tf.summary.scalar("batch KL loss", self.batch_kl_loss)
        tf.summary.scalar("batch annealed KL loss", self.KL_weight_placeholder * self.batch_kl_loss)

        tf.summary.scalar("batch baseline surrogate", self.baseline_surrogate)
        tf.summary.scalar("batch inference surrogate", self.inference_surrogate)
        tf.summary.scalar("batch regularization loss", self.reg_loss)

        tf.summary.scalar("batch log perplexity", self.log_perplexity)

        # self.check_op = tf.add_check_numerics_ops()
        self.check_op = self.x
