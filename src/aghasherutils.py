import os
from scipy.spatial.distance import cdist
import numpy as np


def pdist2(X, Y, metric):
    # scipy has a cdist function that works like matlab's pdist2 function.
    # For square euclidean distance it is slow for the version of scipy you have.
    # For details on its slowness, see https://github.com/scipy/scipy/issues/3251
    # In your tests, it took over 16 seconds versus less than 4 seconds for the
    # implementation below (where X has 69,000 elements and Y had 300).
    # (this has squared Euclidean distances).
    return cdist(X,Y,metric = metric)


def standardize(X):
    # Assumes columns contain variables/features, and rows contain
    # observations/instances.
    means = np.mean(X, 0, keepdims=True)
    stds = np.std(X, 0, keepdims=True)
    return (X - means) / stds
