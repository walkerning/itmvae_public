# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

here = os.path.dirname(os.path.abspath((__file__)))

# meta infos
NAME = "vae_topicmodel"
DESCRIPTION = "VAE topic model"
VERSION = "0.1"

AUTHOR = "foxfi"
EMAIL = "foxdoraame@gmail.com"

# package contents
MODULES = []
PACKAGES = find_packages()

# dependencies
INSTALL_REQUIRES = [
    # "pyyaml==3.12",
    #"tensorflow-gpu==1.2.1",
    # "scikit-learn==0.19.0"
]

# entry points
ENTRY_POINTS = """
[console_scripts]
vae_topic_run=vae_topicmodel.run:main
vae_topic_runtest=vae_topicmodel.run_test:main
vae_image_run=vae_topicmodel.image_run:main
vae_image_runtest=vae_topicmodel.image_run_test:main
"""

def read_long_description(filename):
    path = os.path.join(here, filename)
    if os.path.exists(path):
        return open(path).read()
    return ""

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read_long_description("README.md"),
    author=AUTHOR,
    author_email=EMAIL,

    py_modules=MODULES,
    packages=PACKAGES,

    entry_points=ENTRY_POINTS,
    install_requires=INSTALL_REQUIRES
)
