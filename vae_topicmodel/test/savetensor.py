# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

_scope = None

def populate_argparser(parser, scope):
    global _scope
    _scope = scope
    prefix = "--" + (_scope + "-" if _scope else "")
    parser.add_argument(prefix + "split", default="test", choices=["train", "valid", "test"])
    parser.add_argument(prefix + "name", default=None)
    parser.add_argument(prefix + "axis", default=None, type=int, help="concat along this axis, should be the batch dimension. By default use axis 0 for 2-d tensor and axis 1 for 3-d tensor.")

def do_test(model, reader, args):
    arg_prefix = (_scope + "_" if _scope else "")
    split = getattr(args, arg_prefix + "split")
    axis = getattr(args, arg_prefix + "axis")
    tensor_name = getattr(args, arg_prefix + "name")
    assert tensor_name is not None, "--" + arg_prefix + "name is required" 
    tensor_vs = model.topic_prop_on_dataset(split, tensor=getattr(model, tensor_name), axis=axis)
    np.save(args.load + ".savetensor_{}_{}.npy".format(split, tensor_name), tensor_vs)
