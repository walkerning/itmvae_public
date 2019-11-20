# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import cPickle

import numpy as np
import tensorflow as tf

_scope = None

def populate_argparser(parser, scope):
    global _scope
    _scope = scope
    prefix = "--" + (_scope + "-" if _scope else "")
    parser.add_argument(prefix + "tensors", default=None, help="comma-split names")
    parser.add_argument(prefix + "axis", default=0, type=int, help="concat along this axis, default 0")

def do_test(model, reader, args):
    self = model
    arg_prefix = (_scope + "_" if _scope else "")
    prefix = "--" + (_scope + "-" if _scope else "")
    tensor_names = getattr(args, arg_prefix + "tensors")
    axis = getattr(args, arg_prefix + "axis")
    assert tensor_names, prefix + "tensors must be supplied"
    tensor_names = tensor_names.split(",")
    # save_tensors = [getattr(self, n) for n in tensor_names]
    save_tensors = {n: getattr(self, n) for n in tensor_names}
    save_variables = {n: self.sess.run(v) for n, v in save_tensors.items() if isinstance(v, tf.Variable)}
    tmp_tensor_names, tmp_tensors = zip(*[(n, v) for n, v in save_tensors.items() if not isinstance(v, tf.Variable)])
    
    train_tensors_dct = dict(zip(tmp_tensor_names, [np.concatenate(vs, axis=axis) for vs in zip(*self.topic_prop_on_dataset("train", tensor=tmp_tensors, concat=False))]))
    test_tensors_dct = dict(zip(tmp_tensor_names, [np.concatenate(vs, axis=axis) for vs in zip(*self.topic_prop_on_dataset("test", tensor=tmp_tensors, concat=False))]))
    train_tensors_dct.update(save_variables)
    test_tensors_dct.update(save_variables)
    train_tensors = [train_tensors_dct[n] for n in tensor_names]
    test_tensors = [test_tensors_dct[n] for n in tensor_names]

    # train_tensors = []
    # test_tensors = []
    # for t_name, tensor in zip(tensor_names, save_tensors):
    #     if isinstance(tensor, tf.Variable):
    #         _value = self.sess.run(tensor)
    #         train_tensors.append(_value)
    #         test_tensors.append(_value)
    #     else:
    #         try:
    #             train_tensors.append(self.topic_prop_on_dataset("train", tensor=tensor, axis=0))
    #             test_tensors.append(self.topic_prop_on_dataset("test", tensor=tensor, axis=0))
    #         except ValueError as e:
    #             train_tensors.append(self.topic_prop_on_dataset("train", tensor=tensor, axis=1))
    #             test_tensors.append(self.topic_prop_on_dataset("test", tensor=tensor, axis=1))

    tensor_file = os.path.join(args.load +  "savetensors.pkl")
    with open(tensor_file, "w") as f:
        cPickle.dump({"names": tensor_names,
                      "train": train_tensors,
                      "test": test_tensors
                  }, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print("Save tensors to ", tensor_file)
