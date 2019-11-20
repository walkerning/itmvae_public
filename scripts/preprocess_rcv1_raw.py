# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import glob
import random
random.seed(12345)
import zipfile
import cPickle
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

assert len(sys.argv) == 2, "python preprocess_rcv1_raw result_dir"
res_dir = sys.argv[1]
if not os.path.isdir(res_dir):
    os.mkdir(res_dir)

here = os.path.dirname(os.path.abspath(__file__))
# total: 806791
dire_cd1 = os.path.join(here, "../../rcv1_cd1") # 224 zip files. 473876.
dire_cd2 = os.path.join(here, "../../rcv1_cd2") # 141 zip files. 332915
topic_codes_fname = os.path.join(here, "../../rcv1_cd1/meta/codes/topic_codes.txt")
fname_lst = sorted(glob.glob(os.path.join(dire_cd1, "199*.zip"))) + sorted(glob.glob(os.path.join(dire_cd2, "199*.zip")))
num_zipfile = len(fname_lst)
print("Total {} zip files.".format(num_zipfile))
not_used = set("1POL, 2ECO, 3SPO, 4GEN, 6INS, 7RSK, 8YDB, 9BNX, ADS10, BRP11, ENT12, PRB13, BNW14, G11, G111, G112, G113, G12, G13, G131, G14, GEDU, MEUR".split(", "))
topics = [l.split("\t") for l in open(topic_codes_fname, "r").readlines() if not l.startswith(";") and l.strip() and l.split("\t")[0] not in not_used]
# should be 103 topics
print("Num of topic codes: {}".format(len(topics)))

# number of vocabularies
num_words = 10000
code_index_map = dict([(x, i) for i, (x, _) in enumerate(topics)])
cPickle.dump(code_index_map, open(os.path.join(res_dir, "code_index_map.pkl"), "w"))

# initialize nltk
nltk.download("stopwords")

class LemmaTokenizer(object):
    def __init__(self, stop_words):
        # self.wnl = WordNetLemmatizer()
        self.ps = PorterStemmer()
        self.stop_words = stop_words

    def __call__(self, doc):
        # not punctuation, not total digits, not stop words, length > 2
        doc = [self.ps.stem(word).encode("ascii", errors="ignore").decode("ascii") for word in word_tokenize(doc.lower()) if word.isalnum() and not word.isdigit() and word not in self.stop_words]
        return doc

en_stop = set(stopwords.words("english"))
tokenizer = LemmaTokenizer(en_stop)

# number of occurence of every vocabulary
vocabulary_occurences = {}

temp_dir = tempfile.mkdtemp()
print("temp dir path: ", temp_dir)

for _i, fname in enumerate(fname_lst):
    base_fname = os.path.basename(fname)
    f = zipfile.ZipFile(fname, "r")
    xml_fnames = f.namelist()
    print("\r{:4d}/{:<4d} Handling {}: {} xml files.".format(_i+1, num_zipfile, base_fname,
                                                                                                                          len(xml_fnames)), end="")
    temp_file = open(os.path.join(temp_dir, base_fname + ".txt"), "w")
    for xml_fname in xml_fnames:
        et = ET.fromstring(f.open(xml_fname, "r").read())
        text = " ".join([p.text for p in et.findall("text/p")])
        tokens = tokenizer(text)
        for token in tokens:
            vocabulary_occurences[token] = vocabulary_occurences.get(token, 0) + 1
        labels = [c.attrib["code"] for c in et.findall("metadata/codes/code") if c.attrib["code"] in code_index_map]
        if len(labels) == 0:
            continue
        temp_file.write(" ".join(tokens) + ";" + " ".join(labels) + "\n")
    temp_file.close()

print("")

# Choose voabulary
voc_pairs = sorted(vocabulary_occurences.items(), key=lambda x: x[1], reverse=True)[:num_words]
# lexicographical order
voc_pairs = sorted(voc_pairs, key=lambda x: x[0])
vocab_index_map = {voc: index for index, (voc, _) in enumerate(voc_pairs)}
cPickle.dump(vocab_index_map, open(os.path.join(res_dir, "vocab.pkl"), "w"))

# temp for test
# temp_dir = "/tmp/tmpvebCl5"
# vocab_index_map = cPickle.load(open(os.path.join(res_dir, "vocab.pkl"), "r"))

# number of occurence of each topic
topic_num_lst = np.zeros(len(topics), dtype=np.int32)
whole_tokens_lst = []
whole_labels_lst = []
for _i, fname in enumerate(fname_lst):
    lst = open(os.path.join(temp_dir, os.path.basename(fname) + ".txt"), "r").read().strip().split("\n")
    for line in lst:
        tokenstr, labels = line.split(";")
        tokens = np.array([vocab_index_map[t] for t in tokenstr.split(" ") if t in vocab_index_map], dtype=np.int32)
        if labels.split(" ")[0] == "":
            continue
        if len(tokens) == 0:
            # ignore these documents
            continue
        whole_tokens_lst.append(tokens)
        label_indexes = np.array([code_index_map[l] for l in labels.split(" ")], dtype=np.int32)
        whole_labels_lst.append(label_indexes)
        topic_num_lst[label_indexes] += 1

total_num = len(whole_tokens_lst)
print("After filtering docs without target vocabulary: ", total_num)

test_num = 402207
valid_num = 10000
train_num = total_num - test_num - valid_num # approximately 392207
indices = range(total_num)
random.shuffle(indices)
train_indices = set(indices[:train_num])
valid_indices = set(indices[train_num:train_num+valid_num])
test_indices = set(indices[train_num+valid_num:])

# Backup the indices
cPickle.dump((train_indices, valid_indices, test_indices), open(os.path.join(res_dir, "train_valid_test_indices.pkl"), "w"))

# Dump token/label index to npy file for training.
train_indexes_lst = [whole_tokens_lst[ind] for ind in train_indices]
valid_indexes_lst = [whole_tokens_lst[ind] for ind in valid_indices]
test_indexes_lst = [whole_tokens_lst[ind] for ind in test_indices]
train_labels_lst = [whole_labels_lst[ind] for ind in train_indices]
valid_labels_lst = [whole_labels_lst[ind] for ind in valid_indices]
test_labels_lst = [whole_labels_lst[ind] for ind in test_indices]
train_size = len(train_labels_lst)
print("Split train: {} samples.".format(train_size))
valid_size = len(valid_labels_lst)
print("Split valid: {} samples.".format(valid_size))
test_size = len(test_labels_lst)
print("Split test: {} samples.".format(test_size))
print("Total: {} samples.".format(train_size + valid_size + test_size))
print("topic num list:", topic_num_lst)

np.save(os.path.join(res_dir, "train.txt.npy"), train_indexes_lst)
np.save(os.path.join(res_dir, "valid.txt.npy"), valid_indexes_lst)
np.save(os.path.join(res_dir, "test.txt.npy"), test_indexes_lst)

np.save(os.path.join(res_dir, "train.label.npy"), train_labels_lst)
np.save(os.path.join(res_dir, "valid.label.npy"), valid_labels_lst)
np.save(os.path.join(res_dir, "test.label.npy"), test_labels_lst)
