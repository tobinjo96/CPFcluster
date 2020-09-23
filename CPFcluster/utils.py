import numpy as np
import sys
import os

def chunks(lst, n):
  for i in range(0, len(lst), n):
    yield lst[i:i+n]


def distance_matrix(A):
  return np.sqrt(np.square(A[np.newaxis,:,:]-A[:,np.newaxis,:]).sum(2))

def unique_idx(row):
  return np.unique(row, return_index = True)[1]

def brute_force_knn_par(A, idx, k, blocksz, edge_mat, w_mat):
  mat = distance_matrix(A)
  dismat = np.hstack((w_mat, mat))
  idxmat = np.hstack((edge_mat, np.tile(idx, (len(idx), 1))   ))
  outidx = np.ones(shape = (len(idx), k))*np.inf
  outdis = np.ones(shape = (len(idx), k))*np.inf
  for i in range(len(mat)):
    vals, unidx = np.unique(idxmat[i,:], return_index = True)
    unidx = list(unidx)
    if i+k not in unidx:
      print(idx)
      print(i)
      sys.stdout.flush()
    unidx.remove([i+k])
    outdis[i, 0:min(len(unidx)-1, k)] = np.sort(dismat[i,unidx])[0:min(len(unidx) - 1, k)]
    min_k_indices = [unidx[i] for i in np.argsort(dismat[i,unidx])[0:min(len(unidx) -1, k)]]
    outidx[i, 0:min(len(unidx)-1, k)] = idxmat[i,min_k_indices]
    
  return [outidx, outdis]

def func_star(a_b):
  try:
    return brute_force_knn_par(*a_b)
  except Exception as e:
    raise Exception(e)
