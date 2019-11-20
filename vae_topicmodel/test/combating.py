# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import cPickle

import numpy as np
import tensorflow as tf

def populate_argparser(parser, scope):
    pass

def do_test(model, reader, args):
    # Correlation between decoder output weight norm and the KL(q||p) on test set of each latent dimension
    kl_losses = model.topic_prop_on_dataset(tensor=model.KL_loss_3d, axis=0)
    kl_losses = np.mean(kl_losses, axis=0)
    kernel_output = [x for x in tf.trainable_variables() if "gen_layer_1" in x.op.name and "kernel" in x.op.name]
    assert len(kernel_output) == 1
    kernel_l2_norm = np.sqrt(np.sum(model.sess.run(kernel_output[0]) ** 2, axis=1))
    dump_fpath = os.path.join(os.path.dirname(args.load), "combating.pkl")
    with open(dump_fpath, "w") as f:
        cPickle.dump((kl_losses, kernel_l2_norm), f)
        print("Dump the kl divergence and the output weight norm of each latent dim to {}".format(dump_fpath))
