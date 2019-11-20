# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from .image_vae import ImageVAE
from .gaussian_vae import GaussianVAE
from .gaussian_prior_vae import GaussianPriorVAE
from .dirichlet_process_vae import DirichletProcessVAE
from .dirichlet_process_betarv_vae import DirichletProcessBetaRVVAE

class ImageGaussianVAE(ImageVAE, GaussianVAE):
    MODEL_NAME = "ImageGaussianVAE"

class ImageDPVAE(ImageVAE, DirichletProcessVAE):
    MODEL_NAME = "ImageDirichletProcessVAE"

class ImageDPBetaRVVAE(ImageVAE, DirichletProcessBetaRVVAE):
    MODEL_NAME = "ImageDirichletProcessBetaRVVAE"

class ImageGaussianPriorVAE(ImageVAE, GaussianPriorVAE):
    MODEL_NAME = "ImageGaussianPriorVAE"
