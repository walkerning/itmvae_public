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
    parser.add_argument(prefix + "cols", type=int, default=10)
    parser.add_argument(prefix + "rows", type=int, default=10)

def do_test(model, reader, args):
    arg_prefix = (_scope + "_" if _scope else "")
    cols = getattr(args, arg_prefix + "cols")
    rows = getattr(args, arg_prefix + "rows")
    recons = model.generate(cols * rows)
    recons = recons.reshape((-1, int(np.sqrt(recons.shape[1])), int(np.sqrt(recons.shape[1]))))

    save_fpath = args.load + ".imagegen.png"
    save_plot(recons, save_fpath, shape=(rows, cols))
    print("Save the random generated images from prior to {}".format(save_fpath))
