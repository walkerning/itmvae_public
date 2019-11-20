# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import random
random.seed(12345)
import cPickle

import numpy as np
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from stop_words import get_stop_words

assert len(sys.argv) == 2, "python preprocess_rcv1_raw result_dir"
res_dir = sys.argv[1]
if not os.path.isdir(res_dir):
    os.mkdir(res_dir)

nltk.download("stopwords")

class LemmaTokenizer(object):
    def __init__(self, stop_words):
        self.ps = PorterStemmer()
        self.stop_words = stop_words

    def __call__(self, doc):
        doc = [self.ps.stem(word) for word in word_tokenize(doc.lower()) if word.isalnum() and not word.isdigit() and word not in self.stop_words and len(word) > 2]
        return doc

newsgroups_train = fetch_20newsgroups(subset="train",
                                      remove=["quotes", "headers", "footers"])
newsgroups_test = fetch_20newsgroups(subset="test",
                                     remove=["quotes", "headers", "footers"])
print("20news groups dataset fetched. split train: {} samples; split test: {} samples.".format(len(newsgroups_train["target"]), len(newsgroups_test["target"])))

valid_num = 102
train_num = len(newsgroups_train["target"]) - valid_num
num_words = 2000

train_labels = newsgroups_train["target"]
test_labels = newsgroups_test["target"]

# number of occurence of every vocabulary
vocabulary_occurences = {}

# stop words
en_stop = set(stopwords.words("english"))
tokenizer = LemmaTokenizer(en_stop)
train_texts = [tokenizer(doc) for doc in newsgroups_train["data"]] # 11314
test_texts = [tokenizer(doc) for doc in newsgroups_test["data"]] # 7532
for text in train_texts:
    for token in text:
        vocabulary_occurences[token] = vocabulary_occurences.get(token, 0) + 1

# Choose voabulary
voc_pairs = sorted(vocabulary_occurences.items(), key=lambda x: x[1], reverse=True)[:num_words]
# lexicographical order
voc_pairs = sorted(voc_pairs, key=lambda x: x[0])
vocab_index_map = {voc: index for index, (voc, _) in enumerate(voc_pairs)}
cPickle.dump(vocab_index_map, open(os.path.join(res_dir, "vocab.pkl"), "w"))

indices = range(train_num+valid_num)
random.shuffle(indices)
train_indices = indices[:train_num]
valid_indices = indices[train_num:]

# Dump token/label index to npy file for training.
train_indexes_lst = [np.array([vocab_index_map[t] for t in train_texts[ind] if t in vocab_index_map], dtype=np.int32) for ind in train_indices]
valid_indexes_lst = [np.array([vocab_index_map[t] for t in train_texts[ind] if t in vocab_index_map], dtype=np.int32) for ind in valid_indices]
test_indexes_lst = [np.array([vocab_index_map[t] for t in text if t in vocab_index_map], dtype=np.int32) for text in test_texts]
train_labels_lst = [train_labels[ind] for ind in train_indices]
valid_labels_lst = [train_labels[ind] for ind in valid_indices]
test_labels_lst = test_labels

# Backup the indices
cPickle.dump((train_indices, valid_indices), open(os.path.join(res_dir, "train_valid_indices.pkl"), "w"))

np.save(os.path.join(res_dir, "train.txt.npy"), train_indexes_lst)
np.save(os.path.join(res_dir, "valid.txt.npy"), valid_indexes_lst)
np.save(os.path.join(res_dir, "test.txt.npy"), test_indexes_lst)

np.save(os.path.join(res_dir, "train.label.npy"), train_labels_lst)
np.save(os.path.join(res_dir, "valid.label.npy"), valid_labels_lst)
np.save(os.path.join(res_dir, "test.label.npy"), test_labels_lst)
