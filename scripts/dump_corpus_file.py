# -*- coding: utf-8 -*-

import os
import sys
import cPickle
import itertools
import numpy as np

if len(sys.argv) < 2:
    print "Usage: python dump_corpus_file <dataset>"

dataset = sys.argv[1]
dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "vae_topicmodel/datasets/", dataset)
NUM_PER_FILE = 3000
out_dir = os.path.join(dataset_dir, "./ref_corpus")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# read preprocessed indexes files and vocab file
corpus = itertools.chain(*[np.load(os.path.join(dataset_dir, name + ".txt.npy"), allow_pickle=True) for name in ["train", "valid", "test"]])
vocab = cPickle.load(open(os.path.join(dataset_dir, "vocab.pkl"), "r"))
inverse_vocab = {v:k for k, v in vocab.iteritems()}

# dump clean reference corpus
f_ind = 0
ind = 0
out_fname = os.path.join(out_dir, "corpus_{}.txt".format(f_ind))
out_f = open(out_fname, "w")
print "Write into ", out_fname

for f in corpus:
    # if dataset == "rcv1_v2":
    #     string = " ".join(itertools.chain.from_iterable([inverse_vocab[ind]] * num for (ind, num)
    #                                                     in f if ind in inverse_vocab and num > 0)) + "\n"
    #else:
    string = " ".join(inverse_vocab[ind] for ind in f if ind in inverse_vocab) + "\n"
    out_f.write(string)
    ind += 1
    if ind == NUM_PER_FILE:
        ind = 0
        f_ind += 1
        out_fname = os.path.join(out_dir, "corpus_{}.txt".format(f_ind))
        out_f = open(out_fname, "w")
        print "Write into ", out_fname
    
