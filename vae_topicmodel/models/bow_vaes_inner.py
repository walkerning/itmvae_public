# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from .bow_vae import BoWVAE
from .gaussian_vae import GaussianVAE
from .dirichlet_process_vae import DirichletProcessVAE
from .dirichlet_process_betarv_vae import DirichletProcessBetaRVVAE
from .hierarchical_dirichlet_process_vae import HierarchicalDirichletProcessVAE

class BoWGaussianVAE(BoWVAE, GaussianVAE):
    MODEL_NAME = "BoWGaussianVAE"

class BoWDPVAE(BoWVAE, DirichletProcessVAE):
    MODEL_NAME = "BoWDirichletProcessVAE"

class BoWDPBetaRVVAE(BoWVAE, DirichletProcessBetaRVVAE):
    MODEL_NAME = "BoWDirichletProcessBetaRVVAE"

class BoWHDPVAE(BoWVAE, HierarchicalDirichletProcessVAE):
    MODEL_NAME="BoWHierarchicalDirichletProcessVAE"
