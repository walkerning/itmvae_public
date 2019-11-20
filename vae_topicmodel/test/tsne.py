# -*- coding: utf-8 -*-

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import numpy as np

_scope = None

def populate_argparser(parser, scope):
    global _scope
    _scope = scope
    prefix = "--" + (_scope + "-" if _scope else "")
    parser.add_argument(prefix + "sample_num", type=int, default=None, help="Sample number every class, by default, using the whole test set.")

def do_test(model, reader, args):
    arg_prefix = (_scope + "_" if _scope else "")
    sample_num = getattr(args, arg_prefix + "sample_num")
    labels = reader.load_label_from_type("test")
    topics = np.mean(model.topic_prop_on_dataset(), axis=0)
    avail_classes = np.unique(labels)
    if sample_num is not None:
        indexes = np.concatenate([np.where(labels == cls)[0][:sample_num] for cls in avail_classes])
        topics = topics[indexes]
        labels = labels[indexes]
    print("Start fitting TSNE... sample num per class: {}; Total num: {}".format(sample_num, len(labels)))
    tsne = TSNE(n_components=2, random_state=0)
    embeding = tsne.fit_transform(topics)
    np.save(args.load + ".tsne_{}.npy".format(sample_num), embeding)

    plt.figure(figsize=(6, 6))
    plt.scatter(embeding[:, 0], embeding[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", len(avail_classes)))
    plt.colorbar(ticks=range(len(avail_classes)))
    save_fpath = args.load + ".tsne_{}.png".format(sample_num)
    plt.savefig(save_fpath)
    print("Save the TSNE of latent representation ({} samples each class) to {}".format(sample_num, save_fpath))

