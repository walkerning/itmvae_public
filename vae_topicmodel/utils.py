# -*- coding: utf-8 -*-

from __future__ import print_function

from functools import wraps

def save_plot(x, fname, shape=(10, 10)):
    import numpy as np
    from skimage import io, img_as_ubyte
    num_image = x.shape[0]
    assert len(x.shape) == 3 or len(x.shape) == 4
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=-1)
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    assert np.prod(shape) >= num_image
    height, width = x.shape[1:3]
    ret = np.zeros((height * r, width * c, n_channels), dtype="uint8")
    for i in range(r):
        for j in range(c):
            if i * c + j < num_image:
                ret[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(fname, ret)

def call_once(func):
    @wraps(func)
    def _func(self):
        property_name = "_prop_" + func.__name__
        if not hasattr(self, property_name):
            setattr(self, property_name, func(self))
        return getattr(self, property_name)
    return _func
