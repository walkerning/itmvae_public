# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import glob

here = os.path.dirname(os.path.abspath(__file__))

def is_legal_test_module(mod):
    return hasattr(mod, "populate_argparser") and hasattr(mod, "do_test")

def load_test_modules():
    fpaths = glob.iglob(os.path.join(here, "*.py"))
    for fpath in fpaths:
        if fpath.endswith("__init__.py"):
            continue
        modname = os.path.splitext(os.path.basename(fpath))[0]
        try:
            mod = __import__("vae_topicmodel.test." + modname, fromlist=["*"])
        except ImportError as e:
            print("ERROR: Load test module {} failed: {}".format(modname, e))
            continue
        if is_legal_test_module(mod):
            yield modname, mod
