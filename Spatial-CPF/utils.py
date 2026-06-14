import numpy as np
import sys
import os
from sklearn.metrics.pairwise import euclidean_distances

def chunks(lst, n):
  for i in range(0, len(lst), n):
    yield lst[i:i+n]

def density_broad_search_star(a_b):
  try:
    # sklearn returns shape (n_candidates, 1) here.  Older NumPy versions
    # tolerated assigning one-element rows to scalar array slots; NumPy 2.x
    # correctly rejects that with "setting an array element with a sequence".
    return euclidean_distances(a_b[1], a_b[0]).ravel()
  except Exception as e:
    raise Exception(e)
