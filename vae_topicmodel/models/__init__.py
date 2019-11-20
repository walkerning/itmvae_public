# Default floatX is float32
import tensorflow
import numpy as np
tensorflow.floatX = tensorflow.float32
np.floatX = np.float32

from base import Model
import dmfvi
import HDP
import bow_vaes_inner
import image_vaes_inner
