# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import sys
import time
import copy
import cPickle
import tempfile
from datetime import datetime

import numpy as np
import tensorflow as tf

from .base import Model
from vae_topicmodel.utils import call_once

def _get_shape(t):
    return [int(s) for s in t.get_shape()]

def _log_input(x):
    return np.log(1 + x)

def get_expdecay_value(epoch, cfg):
    return min(cfg.get("max", np.inf), max(cfg.get("min", -np.inf), cfg["start"] * (cfg["rate"] ** max(np.floor((epoch - cfg.get("sepoch", 0)) / cfg["every"]), 0))))

def get_linear_value(epoch, cfg):
    return min(cfg.get("max", np.inf), max(cfg.get("min", -np.inf), cfg["start"] + (cfg["step"] * max(np.floor((epoch - cfg.get("sepoch", 0)) / cfg["every"]), 0))))

def get_reciprocal_value(epoch, cfg):
    return cfg["start"] / (epoch + cfg.get("sepoch", 1))**(cfg["rate"])

def get_schedule_value(epoch, schedule):
    if not isinstance(schedule, dict):
        assert isinstance(schedule, (float, np.float, np.float32))
        return schedule
    type_ = schedule.get("type", "expdecay")
    assert type_ in {"expdecay", "linear", "reciprocal"}
    return globals()["get_" + type_ + "_value"](epoch, schedule)

class VAE(Model):
    """
    Variational Auto-Encoder implemented with Tensorflow
    """
    _default_train_cfg = {
        "max_epochs": 100,
        "batch_size": 200,

        # Optimizer configuration
        "optimizer": "AdamOptimizer",
        "optimizer_cfg": {
            "learning_rate": 0.002,
            "beta1": 0.8
        },
        "dropout_keep_prob": 1,

        "encoder_optimizer": "AdamOptimizer",
        "encoder_optimizer_cfg": {
            "learning_rate": 0.002,
            "beta1": 0.8
        },

        # KL annealing configuration
        "kl_annealing": {
            "start": 0, # start KL weight
            "interval": 1, # epoch
            "step": 0.05, # by default, 20 steps to reach 1
            "max": 1 # end KL weight
        },

        # KL clip method as in paper `Improved Variational Inference with Inverse Autoregressive Flow`
        "kl_min": None,

        # print configuration
        "print_every": 5,
        "print_tensor_names": [], # additional print tensor names(average in epoch)

        # preprocessing
        "log_input": False,

        # Early stopping. by default, no early stopping is used. set `stop_threshold` to
        # a smaller value, and `load_best_and_test` to True to enable it.
        # See `configs/20news/dirichlet_process` for an example.
        "check_early_stop_every": 50,
        "load_best_and_test": False,
        # If the validation indicator(log perplexity for topic model or loss for image)
        # do not drop for `stop_threshold` epochs, stop the training.
        # Or if `learning_rate_decay` is not None, decay the learning rate
        "stop_threshold": 2000,
        "lr_threshold": None, # minimal lr threshold
        "check_decay_threshold": 25, # Must be a division of stop threshold
        "learning_rate_decay": None,
        "lr_not_decay_epoch_threshold": None,

        # For test
        # "test_batch_size": None,
        "test_batch_size": 200,
        "test_likelihood_method": "IS", # one of IS, AIS

        # Snapshot
        "snapshot_every": None,
        "snapshot_dir": None,

        # Save tensors
        "save_tensor_every": None,
        "save_tensor_dir": None,
        "save_tensor_names": [],

        # Diversity
        "diversity_weight": 0.1,

        # For learning encoder/decoder alternatively
        "use_alternating": False,
        "use_beta_alternating": False,

        # For prior_beta learning
        "use_natural_gradient": False,
        "sgd_start_epoch": 0,
        "use_alternating_betarv": False,

        # Type of reconstruction error for image
        "image_loss_type": "binary",

        # Data shuffling
        "shuffle": False,
        "batch_norm_train_epoch": None,

        "schedule": {},

        # Continue training
        "start_epoch": 1
    }

    def __init__(self, cfg, train_cfg):
        super(VAE, self).__init__(cfg, train_cfg)

        self._sess = None # require a tensorflow session
        # self.KL_weight_placeholder = tf.placeholder_with_default(tf.constant(1, dtype=tf.floatX), [], "KL_weight_placeholder")
        self.KL_weight_placeholder = tf.placeholder_with_default(tf.constant(train_cfg.get("test_kl_weight", 1), dtype=tf.floatX), [], "KL_weight_placeholder")
        # for now
        self.training_placeholder = tf.placeholder(tf.bool, name="training_placeholder")
        self.keep_prob_placeholder = tf.placeholder_with_default(tf.constant(1., dtype=tf.floatX), [], name="keep_prob_placeholder")
        self.dataset_size_placeholder = tf.placeholder_with_default(tf.constant(self.reader.train_size, dtype=tf.floatX), [], name="dataset_size_placeholder")
        self.scope = cfg.get("scope", self.__class__.__name__)

        self.opt_surrogate = None
        self.theta = None
        self.loss = None
        self.saver = None
        self.summary_merged = None

        self.train_cfg = copy.deepcopy(self._default_train_cfg)
        self.train_cfg.update(train_cfg)
        self.diversity_weight_placeholder = tf.placeholder_with_default(tf.constant(self.train_cfg["diversity_weight"], dtype=tf.floatX), [], "diversity_weight_placeholder")

        self.cfg = copy.deepcopy(self._default_cfg)
        self.cfg.update(cfg)
        print("{}: Use model configration:\n{}\n\nUse train configuration:\n{}".format(self.MODEL_NAME, "\n".join("\t{:30}: {}".format(k, v) for k, v in sorted(self.cfg.iteritems(), key=lambda item: item[0]) if k in self._default_cfg),
                                                                                              "\n".join("\t{:30}: {}".format(k, v) for k, v in sorted(self.train_cfg.iteritems(), key=lambda item: item[0]))))

        self.cfg = {k: (self._cfg_handlers.get(k, lambda c: c)(v)) for k, v in self.cfg.iteritems()}

        self.x = tf.placeholder(tf.floatX, [None, self.vocab_dim])
        self.x_input = self.x
        if self.cfg["encoder_log_input"]:
            print("Log input in encoder...")
            self.x_input = tf.log(self.x_input + 1)
        # if self.cfg["centering_input"]: # DO NOT USE FOR NOW
        #     self.x_input = self.x_input - self.reader.mean()
        with tf.variable_scope(self.scope):
            self.inference_out = self.build_inference_net(self.x_input)
            z = self.build_stochastic_layer(self.inference_out)
            self.build_gen_net(z)
            self.build_loss()

            self.build_test_gen_net() # A generator from z placeholder

    @property
    def sess(self):
        """
        Create session on demand, and reuse it.
        """
        if self._sess:
            return self._sess
        self._sess = self.cfg.get("sess", None)
        if self._sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self._sess = tf.Session(config=config)
        return self._sess

    def init(self, test_only=False):
        if not test_only:
            # Build the the optimize surrogate
            kl_surrogate = self.KL_weight_placeholder * self.batch_kl_loss
            if self.train_cfg["kl_min"] is not None:
                kl_surrogate = tf.maximum(kl_surrogate, self.train_cfg["kl_min"])
            self.opt_surrogate = self.batch_rec_loss + kl_surrogate + self.reg_loss
            if hasattr(self, "diversity_loss"):
                self.opt_surrogate += self.diversity_weight_placeholder * self.diversity_loss

            # Construct optimize tensor
            self.learning_rate = tf.placeholder(tf.floatX, [])
            self.encoder_learning_rate = tf.placeholder(tf.floatX, [])
            self.learning_rate_v = self.train_cfg["optimizer_cfg"]["learning_rate"]
            if self.train_cfg["learning_rate_decay"] is None:
                self.optimizer = getattr(tf.train, self.train_cfg["optimizer"])(**self.train_cfg["optimizer_cfg"])
            else:
                self.train_cfg["optimizer_cfg"]["learning_rate"] = self.learning_rate
                self.optimizer = getattr(tf.train, self.train_cfg["optimizer"])(**self.train_cfg["optimizer_cfg"])
            self.encoder_learning_rate_v = self.train_cfg["encoder_optimizer_cfg"]["learning_rate"]
            if self.train_cfg["use_alternating"] or self.train_cfg["use_beta_alternating"]:
                if self.train_cfg["encoder_learning_rate_decay"] is None:
                    self.encoder_optimizer = getattr(tf.train, self.train_cfg["encoder_optimizer"])(**self.train_cfg["encoder_optimizer_cfg"])
                else:
                    self.train_cfg["encoder_optimizer_cfg"]["learning_rate"] = self.encoder_learning_rate
                    self.encoder_optimizer = getattr(tf.train, self.train_cfg["encoder_optimizer"])(**self.train_cfg["encoder_optimizer_cfg"])

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            t_vars = tf.trainable_variables()
            adam_vars = [var for var in t_vars if "sgd" not in var.name]
            sgd_vars = [var for var in t_vars if "sgd" in var.name]
            if self.train_cfg.has_key("sgd_lr") and self.train_cfg["sgd_lr"] > 0:
                print("SGD optimized vars: {}; learning rate: {}".format(sgd_vars, self.train_cfg["sgd_lr"]))
            elif sgd_vars:
                print("WARNING: sgd variables {} not being optimized! specify `sgd_lr` traininig config to optimize!".format(sgd_vars))
            with tf.control_dependencies(update_ops):
                if self.train_cfg["use_alternating"]:
                    encoder_vars = [var for var in adam_vars if "inference" in var.name]
                    other_vars = [var for var in adam_vars if "inference" not in var.name]
                    self.opt_step = tf.group(self.optimizer.minimize(self._optimize_surrogate,
                                                                     var_list=other_vars),
                                             self.encoder_optimizer.minimize(self._optimize_surrogate,
                                                                             var_list=encoder_vars))
                elif self.train_cfg["use_beta_alternating"]:
                    other_vars = [var for var in adam_vars if "beta" in var.op.name and "batch" not in var.op.name]
                    encoder_vars = [var for var in adam_vars if "beta" not in var.op.name or "batch" in var.op.name]
                    self.opt_step = tf.group(self.optimizer.minimize(self._optimize_surrogate,
                                                                     var_list=other_vars),
                                             self.encoder_optimizer.minimize(self._optimize_surrogate,
                                                                             var_list=encoder_vars))
                else:
                    self.opt_step = self.optimizer.minimize(self._optimize_surrogate,
                                                            var_list=adam_vars)                    

                if self.train_cfg.has_key("sgd_lr"):
                    self.sgd_optimizer = tf.train.GradientDescentOptimizer(self.train_cfg["sgd_lr"])
                    grads_and_vars = self.sgd_optimizer.compute_gradients(self._optimize_surrogate, var_list=sgd_vars)
                    # self.opt_step_sgd = .minimize(self._optimize_surrogate,
                    #                                                                                          var_list=sgd_vars)
                    if self.train_cfg["use_natural_gradient"]:
                        _tmp = tf.digamma(self.post_gamma1)
                        alpha_didigamma = tf.gradients(_tmp, self.post_gamma1)[0]
                        beta_sq = tf.square(self.post_gamma2)
                        # Exp
                        # alpha_old_grad = grads_and_vars[0][0] / self.post_gamma1
                        # alpha_old_grad = grads_and_vars[1][0] / self.post_gamma2
                        # Softplus
                        alpha_old_grad = grads_and_vars[0][0] * (1 + tf.exp(-self.log_post_gamma1))
                        beta_old_grad = grads_and_vars[1][0] * (1 + tf.exp(-self.log_post_gamma2))

                        # Using tf.matrix_inverse
                        lognorm_hessian = tf.stack([tf.stack([alpha_didigamma, -1/self.post_gamma2]), tf.stack([-1/self.post_gamma2, self.post_gamma1/beta_sq])])
                        alpha_new_grad, beta_new_grad = tf.unstack(tf.squeeze(tf.matmul(tf.matrix_inverse(lognorm_hessian), tf.expand_dims(tf.stack([alpha_old_grad, beta_old_grad]), axis=-1))))
                        # # For exp
                        # log_alpha_new_grad = alpha_new_grad * self.post_gamma1
                        # log_beta_new_grad = beta_new_grad * self.post_gamma2
                        # For softplus
                        log_alpha_new_grad = alpha_new_grad / (1 + tf.exp(-self.log_post_gamma1))
                        log_beta_new_grad = beta_new_grad / (1 + tf.exp(-self.log_post_gamma2))

                        new_grads_and_vars = [(log_alpha_new_grad, grads_and_vars[0][1]), (log_beta_new_grad, grads_and_vars[1][1])]
                    else:
                        new_grads_and_vars = grads_and_vars
                    self.opt_step_sgd = self.sgd_optimizer.apply_gradients(new_grads_and_vars)
                    self.opt_step_group = tf.group(self.opt_step, self.opt_step_sgd)

        self.sess.run(tf.global_variables_initializer())

        if not test_only and self.cfg["summary_dir"]:
            self.summary_writer = tf.summary.FileWriter(self.cfg["summary_dir"], self.sess.graph)
            self.summary_merged = tf.summary.merge_all()
        else:
            self.summary_writer = None

    def build_inference_net(self, x):
        inet_structure = self.cfg["inference_net_structure"]
        layer = x
        for i, dim in enumerate(inet_structure):
            layer = tf.layers.dense(layer, dim, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=(tf.contrib.layers.l2_regularizer(scale=self.cfg["inference_regularizer"]) 
                                                        if self.cfg["inference_regularizer"] else None),
                                    name="inference_layer_{}".format(i+1))
            # FIXME: batch norm should not be in cfg... it should be in training cfg... it's in fact not part of model...
            if self.cfg["batch_norm_inference_net"]:
                layer = tf.contrib.layers.batch_norm(layer, is_training=self.training_placeholder, scope="inference_layer_batchnorm_{}".format(i+1), renorm=self.cfg.get("batch_renorm", False))
            layer = self.cfg["transfer_fct"](layer)
        if self.cfg["dropout_inference_out"]:
            layer = tf.nn.dropout(layer, self.keep_prob_placeholder)

        return layer

    def build_test_gen_net(self):
        layer = self.z_placeholder = tf.placeholder(tf.floatX, [None, self.topic_dim])
        gnet_structure = self.cfg["gen_net_structure"]
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for i, dim in enumerate(gnet_structure):
                layer = tf.layers.dense(layer, dim, activation=None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.zeros_initializer(), name="gen_layer_{}".format(i+1))
                if self.cfg["batch_norm_gen_net"]:
                    layer = tf.contrib.layers.batch_norm(layer, is_training=False, renorm=self.cfg.get("batch_renorm", False))
                layer = self.cfg["transfer_fct"](layer)
            theta = layer
            if self.cfg["softmax_topic_vector"]:
                theta = tf.nn.softmax(theta)
            self.test_w_dist = self._get_collapsed_word_dist(theta, test=True)

    def build_gen_net(self, z):
        gnet_structure = self.cfg["gen_net_structure"]
        layer = z
        for i, dim in enumerate(gnet_structure):
            layer = tf.layers.dense(layer, dim, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(), name="gen_layer_{}".format(i+1))
            if self.cfg["batch_norm_gen_net"]:
                layer = tf.contrib.layers.batch_norm(layer, is_training=self.training_placeholder)
            layer = self.cfg["transfer_fct"](layer)
        self.theta = layer
        if self.cfg["softmax_topic_vector"]:
            self.theta = tf.nn.softmax(self.theta)
        if self.cfg["dropout_topic_vector"]:
            self.theta = tf.nn.dropout(self.theta, self.keep_prob_placeholder)
        self.w_dist = self._get_collapsed_word_dist(self.theta)

    def build_loss(self):
        self.build_rec_loss() # set self.batch_rec_loss
        self.build_kl_loss() # set self.batch_kl_loss
        tf.summary.scalar("batch KL loss", self.batch_kl_loss)
        self.loss = self.batch_rec_loss + self.batch_kl_loss

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = tf.add_n(reg_losses) if reg_losses else tf.constant(0., dtype=tf.floatX)
        self._init_loss()

    def train(self):
        self.best_model_tmp_dir = tempfile.mkdtemp()
        self.best_model_path = os.path.join(self.best_model_tmp_dir, "best")
        if self.train_cfg["log_input"]:
            gen = self.reader.iterator(self.train_cfg["batch_size"], shuffle=self.train_cfg["shuffle"], func=_log_input)
        else:
            gen = self.reader.iterator(self.train_cfg["batch_size"], shuffle=self.train_cfg["shuffle"])
        valid_loss = None
        test_loss = None
        best_test_loss = None
        best_valid_indicator_epoch = 0
        steps_per_epoch = self.reader.train_size // self.train_cfg["batch_size"]
        total_iter = 0
        # annealing_cfg = self.train_cfg["kl_annealing"]
        def _getattr_or_eval(s, n):
            t = getattr(self, n, None)
            if t is None:
                t = eval("self." + n)
            return t
        print_tensors = [_getattr_or_eval(self, n) for n in self.train_cfg["print_tensor_names"]]
        save_tensors = [getattr(self, n) for n in self.train_cfg["save_tensor_names"]]
        with self.sess.as_default():
            for epoch in xrange(self.train_cfg.get("start_epoch", 1), self.train_cfg["max_epochs"] + 1):
                # kl_weight_v = min(annealing_cfg["max"],
                #                   annealing_cfg["start"] + (epoch -1) // annealing_cfg["interval"] * \
                #                   annealing_cfg["step"])
                loss_v_avg = 0
                indicator_v_avg = 0
                surrogate_v_avg = 0
                # additional_v_avg = np.zeros([len(self.train_cfg["print_tensor_names"])])
                additional_v_avg = [np.zeros(_get_shape(t)) for t in print_tensors]
                schedule_info = ""
                schedule_dict = {}
                # for simple profiling
                gen_time = 0
                run_time = 0
                for attr_name, schedule in self.train_cfg["schedule"].iteritems():
                    v = get_schedule_value(epoch, schedule)
                    schedule_dict[getattr(self, attr_name)] = v
                    schedule_info += "{:10}: {:.5f}; ".format(attr_name, v)
                for step in xrange(1, steps_per_epoch + 1):
                    gen_start_time = time.time()
                    x_v = gen.next()
                    gen_time += time.time() - gen_start_time
                    feed_dict = {
                        self.x: x_v,
                        self.learning_rate: self.learning_rate_v,
                        self.encoder_learning_rate: self.encoder_learning_rate_v,
                        # self.KL_weight_placeholder: kl_weight_v, # USE schedule
                        self.training_placeholder: False if (self.train_cfg["batch_norm_train_epoch"] is not None and epoch > self.train_cfg["batch_norm_train_epoch"]) else True,
                        self.keep_prob_placeholder: self.train_cfg["dropout_keep_prob"]
                    }
                    feed_dict.update(schedule_dict)
                    if not self.train_cfg["use_alternating_betarv"] and self.train_cfg.has_key("sgd_lr") and epoch > self.train_cfg["sgd_start_epoch"]:
                        opt_step = self.opt_step_group
                    else:
                        opt_step = self.opt_step
                    run_start_time = time.time()
                    if self.train_cfg["use_alternating_betarv"] and epoch > self.train_cfg["sgd_start_epoch"]:
                        _ = self.sess.run(self.opt_step_sgd, feed_dict=feed_dict)
                    if self.summary_merged is not None and self.summary_writer:
                        _, summary, loss_v, indicator_v, surrogate_v, additional_v = self.sess.run([opt_step, self.summary_merged,
                                                                                                    self.loss, self._test_tensor,
                                                                                                    self._optimize_surrogate, print_tensors],
                                                                        feed_dict=feed_dict)
                        self.summary_writer.add_summary(summary, total_iter)
                    else:
                        _, loss_v, indicator_v, surrogate_v, additional_v = self.sess.run([opt_step, self.loss, self._test_tensor,
                                                                                           self._optimize_surrogate, print_tensors],
                                                                                          feed_dict=feed_dict)
                    run_time += time.time() - run_start_time

                    loss_v_avg += loss_v
                    indicator_v_avg += indicator_v
                    surrogate_v_avg += surrogate_v
                    # additional_v_avg += np.array(additional_v)
                    additional_v_avg = [avg + v for avg, v in zip(additional_v_avg, additional_v)]
                    total_iter += 1
                    print("\rEpoch {:4}/{:<4}: Training ... {:5}/{:<5}".format(epoch, self.train_cfg["max_epochs"], step, steps_per_epoch),
                          end="", file=sys.stderr)
                gen_time /= steps_per_epoch
                run_time /= steps_per_epoch
                loss_v_avg /= steps_per_epoch
                indicator_v_avg /= steps_per_epoch
                surrogate_v_avg /= steps_per_epoch
                # additional_v_avg /= steps_per_epoch
                additional_v_avg = [v / steps_per_epoch for v in additional_v_avg]
                print("\r", end="", file=sys.stderr)
                if epoch < 5 or epoch % self.train_cfg["print_every"] == 0:
                    print("{}: Epoch {:4}/{:<4}: loss (average): {:10.5f}; {}: {:10.5f}; perp: {:10.5f}; surrogate (average): {:10.5f}. gen time: {:.3f} sec/batch; run time: {:.3f} sec/batch\n\t{}\n\t{}"\
                          .format(datetime.now(), epoch, self.train_cfg["max_epochs"], loss_v_avg, self._test_tensor_name, indicator_v_avg, np.exp(indicator_v_avg), surrogate_v_avg, 
                                  gen_time, run_time,
                                  schedule_info,
                                  "\n\t".join(["{:20} : {:10}".format(name, v) for name, v in zip(self.train_cfg["print_tensor_names"],
                                                                                                  additional_v_avg)])))
                self.on_epoch_end(epoch, schedule_dict)
                if epoch % self.train_cfg["check_early_stop_every"] == 0:
                    indicator_v = self.test_on_dataset(data_type="valid")
                    test_indicator_v = self.test_on_dataset(data_type="test")
                    if valid_loss is None or indicator_v < valid_loss:
                        valid_loss = indicator_v
                        best_valid_indicator_epoch = epoch
                        self.save(self.best_model_path)
                    if best_test_loss is None or test_indicator_v < best_test_loss:
                        best_test_loss = test_indicator_v
                        best_test_indicator_epoch = epoch
                    print("\tValid set indicator({}): {:10.5f}; Best (epoch {:3}): {:10.5f}; perp: {:10.5f}".format(
                        self._test_tensor_name,
                        indicator_v,
                        best_valid_indicator_epoch,
                        valid_loss,
                        np.exp(valid_loss)))
                    print("\tTest set indicator({}): {:10.5f}; Best (epoch {:3}): {:10.5f}; perp: {:10.5f}".format(
                        self._test_tensor_name,
                        test_indicator_v,
                        best_test_indicator_epoch,
                        best_test_loss,
                        np.exp(best_test_loss)))
                    if epoch - best_valid_indicator_epoch >= self.train_cfg["stop_threshold"]:
                        # decay learning rate
                        print("NOTE: Early stop the training process,"
                              " as the validation performance indicator do not drop for {} epochs.".format(epoch - best_valid_indicator_epoch))    
                        break
                        
                    if epoch >= 200 and epoch % self.train_cfg["check_decay_threshold"] == 0:
                        if self.train_cfg["learning_rate_decay"] is not None:
                            if not (self.train_cfg["lr_not_decay_epoch_threshold"] and epoch >= self.train_cfg["lr_not_decay_epoch_threshold"]):
                                self.learning_rate_v *= self.train_cfg["learning_rate_decay"]
                                print("Decay learning rate to {}".format(self.learning_rate_v))
                            if self.train_cfg["lr_threshold"] is not None and self.learning_rate_v < self.train_cfg["lr_threshold"]:
                                # Reach the minimal learning rate threshold
                                break
                            if self.train_cfg["use_alternating"] and self.train_cfg["encoder_learning_rate_decay"] is not None:
                                self.encoder_learning_rate_v *= self.train_cfg["encoder_learning_rate_decay"]
                                print("Decay encoder learning rate to {}".format(self.encoder_learning_rate_v))

                        # if epoch - best_valid_indicator_epoch >= self.train_cfg["check_decay_threshold"]:


                if self.train_cfg["snapshot_every"] and self.train_cfg["snapshot_dir"] \
                   and epoch % self.train_cfg["snapshot_every"] == 0:
                    snapshot_fname = os.path.join(self.train_cfg["snapshot_dir"], str(epoch))
                    self.save(snapshot_fname)
                    print("Snapshot model to {}".format(snapshot_fname))
                    # NOTE: just for quick debugging.
                    if hasattr(self, "_topic_components_tensor"):
                        self.print_top_words(file=open(snapshot_fname + ".txt", "w"))
                    test_loss = self.test_on_dataset(data_type="test")
                    print("\tTest set indicator({}): {:10.5f}".format(self._test_tensor_name, test_loss))
                    open(snapshot_fname + ".loss", "w").write(str(test_loss) + "\n")
                if len(self.train_cfg["save_tensor_names"]) and self.train_cfg["save_tensor_every"] \
                   and self.train_cfg["save_tensor_dir"] \
                   and epoch % self.train_cfg["save_tensor_every"] == 0:
                    test_loss = self.test_on_dataset(data_type="test")
                    train_tensors = []
                    test_tensors = []
                    for t_name, tensor in zip(self.train_cfg["save_tensor_names"], save_tensors):
                        if isinstance(tensor, tf.Variable):
                            _value = self.sess.run(tensor)
                            train_tensors.append(_value)
                            test_tensors.append(_value)
                        else:
                            train_tensors.append(self.topic_prop_on_dataset("train", tensor=tensor, axis=0))
                            # valid_tensors = self.topic_prop_on_dataset("valid", tensor=tensor)
                            test_tensors.append(self.topic_prop_on_dataset("test", tensor=tensor, axis=0))

                    tensor_file = os.path.join(self.train_cfg["save_tensor_dir"], "{}.pkl".format(epoch))
                    # np.savez(tensor_file, names=self.train_cfg["save_tensor_names"], train=train_tensors, test=test_tensors)
                    with open(tensor_file, "w") as f:
                        cPickle.dump({"names": self.train_cfg["save_tensor_names"],
                                      "train": train_tensors,
                                      "test": test_tensors
                                  }, f, protocol=cPickle.HIGHEST_PROTOCOL)
                    print("\tTest set perlexity: {:10.5f}; save tensors {} to file {}.".format(test_loss, 
                                                                                               self.train_cfg["save_tensor_names"],
                                                                                               tensor_file))

        if self.train_cfg["load_best_and_test"]:
            print("Load the model of epoch {}...".format(best_valid_indicator_epoch))
            self.load(self.best_model_path)

    def test_on_dataset(self, data_type="test", tensor=None, use_sum=False, random=False, maxstep=None):
        if tensor is None:
            tensor = self._test_tensor
        func=(_log_input if self.train_cfg["log_input"] else lambda x: x)
        _feed_dct = {
            self.training_placeholder: False,
            self.dataset_size_placeholder: getattr(self.reader, data_type + "_size")
        }
        if self.train_cfg["test_batch_size"] is None:
            test_data = self.reader.get_parsed_data_from_type(data_type, func)
            # Total loss / total count in the held-out test set.
            res = self._run(tensor, test_data, _feed_dct)
        else:
            gen = self.reader.iterator_one_pass(self.train_cfg["test_batch_size"], data_type=data_type, func=func, random=random)
            # res = np.zeros(len(tensor) if isinstance(tensor, (list, tuple)) else 1)
            is_lst = isinstance(tensor, (list, tuple))
            res = [np.zeros(_get_shape(t)) for t in tensor] if is_lst else np.zeros(_get_shape(tensor))
            if not is_lst and not res.shape:
                res = 0.0
            num_test = 0
            step = 0
            while 1:
                try:
                    datum = gen.next()
                except StopIteration:
                    break
                batch_mean = np.array(self._run(tensor, datum, _feed_dct))
                if not is_lst:
                    res += batch_mean * len(datum)
                else:
                    res = [r + b * len(datum) for b, r in zip(batch_mean, res)]
                num_test += len(datum)
                print("\rTesting ... {:5}".format(num_test), end="", file=sys.stderr)
                step += 1
                if maxstep and step >= maxstep:
                    break
            if not use_sum:
                res /= num_test
            print("\r", end="", file=sys.stderr)
            print("Finish test {:5} samples".format(num_test))
        #return res if isinstance(tensor, (list, tuple)) else res[0]
        return res

    def topic_prop_on_dataset(self, data_type="test", tensor=None, axis=None, concat=True):
        if tensor is None:
            tensor = self._topic_prop_tensor
        func=(_log_input if self.train_cfg["log_input"] else lambda x: x)
        _feed_dct = {
            self.training_placeholder: False,
            self.dataset_size_placeholder: getattr(self.reader, data_type + "_size")
        }
        if self.train_cfg["test_batch_size"] is None:
            test_data = self.reader.get_parsed_data_from_type(data_type, func)
            # Total loss / total count in the held-out test set.
            res = self._run(tensor, test_data, _feed_dct)
        else:
            gen = self.reader.iterator_one_pass(self.train_cfg["test_batch_size"], data_type=data_type, func=func)
            num_test = 0
            res_lst = []
            while 1:
                try:
                    datum = gen.next()
                except StopIteration:
                    break
                batch_topic_prop = self._run(tensor, datum, _feed_dct)
                res_lst.append(batch_topic_prop)
                num_test += len(datum)
                print("\rTesting ... {:5}".format(num_test), end="", file=sys.stderr)
            if concat:
                if axis is None:
                    # By default, if the tensor is a 3-d tensor, regard the first dimension as MC sampling dimension
                    # concatenate using axis 1 (batch dimension). Otherwise, concatenate using axis 0.
                    if len(tensor.get_shape()) == 3:
                        axis = 1
                    else:
                        axis = 0
                res = np.concatenate(res_lst, axis=axis)
            else:
                res = res_lst
            print("\r", end="", file=sys.stderr)
            print("Finish test {:5} samples".format(num_test))
        return res

    def topic_prop(self, x, feed_dct={}, tensor=None):
        func = (_log_input if self.train_cfg["log_input"] else lambda x: x)
        func_x = func(x)
        _feed_dct = {
            self.training_placeholder: False,
            self.dataset_size_placeholder: getattr(self.reader, data_type + "_size")
        }
        _feed_dct.update(feed_dct)
        if tensor is None:
            tensor = self._topic_prop_tensor
        return self._run(tensor, func_x, _feed_dct)

    def log_likelihood_on_dataset(self, data_type="test"):
        print("Test marginal log likelihood using number of mc samples: {}; method: {}".format(self.cfg["MC_samples"],
                                                                                               self.train_cfg["test_likelihood_method"]))
        return self.test_on_dataset(data_type, tensor=self.log_likelihood_tensor)

    def log_likelihood(self, x):
        _feed_dct = {
            self.training_placeholder: False
        }
        return self._run(self.log_likelihood_tensor, x, _feed_dct)

    def _run(self, target, x, feed_dct={}):
        feed_dict = {self.x: np.expand_dims(x, axis=0) if len(x.shape) == 1 else x}
        feed_dict.update(feed_dct)
        return self.sess.run(target, feed_dict=feed_dict)

    def generate(self, num):
        # Generate from prior
        latent = self.sample_from_prior(num)
        recon = self.sess.run(self.test_w_dist, feed_dict={self.z_placeholder: latent,
                                                           self.training_placeholder: False})
        return recon

    @property
    @call_once
    def log_likelihood_tensor(self):
        if self.train_cfg["test_likelihood_method"] == "IS":
            # Importance sampling to estimate margial likelihood
            log_weighted_p = -self.rec_loss + self.log_prior_pdf - self.log_posterior_pdf

            base = tf.reduce_max(log_weighted_p, axis=0, keep_dims=True) 
            log_weighted_p = log_weighted_p - base
            weighted_p = tf.exp(log_weighted_p)
            p_xi = tf.reduce_mean(weighted_p, axis=0)# / tf.reduce_sum(weight, axis=0)
            _log_likelihood = tf.reduce_mean(tf.check_numerics(tf.log(p_xi), "check_log_p_xi") + base)

        elif self.train_cfg["test_likelihood_method"] == "AIS":
            # Annealed importance sampling to estimate margial likelihood
            pass
        return _log_likelihood

    @property
    def _optimize_surrogate(self):
        return self.opt_surrogate

    @property
    def _topic_prop_tensor(self):
        return None

    @property
    def _test_tensor(self):
        return None

    def save(self, path):
        self.saver = self.saver or tf.train.Saver(max_to_keep=None)
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver = self.saver or tf.train.Saver(max_to_keep=None)
        self.saver.restore(self.sess, path)

    @property
    def topic_components(self):
        return self.sess.run(self._topic_components_tensor)

    def on_epoch_end(self, epoch, schedule_dict):
        # The hook point that will be called on every epoch end
        pass
