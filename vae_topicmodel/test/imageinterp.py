# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import numpy as np

from vae_topicmodel.utils import save_plot

_scope = None

def populate_argparser(parser, scope):
    global _scope
    _scope = scope
    prefix = "--" + (_scope + "-" if _scope else "")
    parser.add_argument(prefix + "labels", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-split list of labels.")
    parser.add_argument(prefix + "interp_num", type=int, default=10)

def do_test(model, reader, args):
    arg_prefix = (_scope + "_" if _scope else "")
    labels = [int(x) for x in getattr(args, arg_prefix + "labels").split(",")]
    interp_num = getattr(args, arg_prefix + "interp_num")
    recons = []
    for label in labels:
        ind1, ind2 = np.where(reader.train_labels == label)[0][:2]
        latent_1 = np.mean(model.topic_prop(reader.train_data[ind1]), axis=0)
        latent_2 = np.mean(model.topic_prop(reader.train_data[ind2]), axis=0)
        interp_ratio1 = np.array(range(interp_num), dtype=np.floatX) / (interp_num-1)
        interp_ratio2 = np.array(list(reversed(interp_ratio1)))
        interp_ratio1 = np.expand_dims(interp_ratio1, axis=-1)
        interp_ratio2 = np.expand_dims(interp_ratio2, axis=-1)

        latent = interp_ratio1 * np.repeat(latent_1, interp_num, axis=0) + \
                 interp_ratio2 * np.repeat(latent_2, interp_num, axis=0)
        recon = model.sess.run(model.test_w_dist, feed_dict={model.z_placeholder: latent,
                                                             model.training_placeholder: False})
        recons.append(recon)

    recons = np.concatenate(recons, axis=0)
    recons = recons.reshape((-1, int(np.sqrt(recons.shape[1])), int(np.sqrt(recons.shape[1]))))

    save_fpath = args.load + ".imageinterp.png"
    save_plot(recons, save_fpath, shape=(len(labels), interp_num))
    print("Save the interpolation images of two latent representation of the same class to {}".format(save_fpath))
