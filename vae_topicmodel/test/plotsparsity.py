# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

def populate_argparser(parser, scope):
    pass

def do_test(model, reader, args):
    # The sparsity of the latent rep. use mc_sample = 1
    latent = model.topic_prop_on_dataset(tensor=model.theta, axis=0)
    latent_avg = np.mean(kl_losses, axis=0)
    plt.figure(figsize=(6,6))
    plt.plot(len(latent_avg), np.log(np.sort(latent_avg)))
    save_fpath = args.load + ".sparsity.png"
    plt.savefig(save_fpath)
    print("Save the plot of sparsity(AVITM) latent representation to {}".format(save_fpath))
