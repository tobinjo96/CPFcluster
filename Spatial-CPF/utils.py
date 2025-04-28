import numpy as np
import sys
import os
from sklearn.metrics.pairwise import euclidean_distances

def chunks(lst, n):
  for i in range(0, len(lst), n):
    yield lst[i:i+n]

def density_broad_search_star(a_b):
  try:
    return euclidean_distances(a_b[1],a_b[0])
  except Exception as e:
    raise Exception(e)
