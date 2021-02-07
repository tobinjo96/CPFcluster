import numpy as np
import sys
import os
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances

def chunks(lst, n):
  for i in range(0, len(lst), n):
    yield lst[i:i+n]

def skip_diag_strided(A):
  m = A.shape[0]
  strided = np.lib.stride_tricks.as_strided
  s0,s1 = A.strides
  return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)

def distance_matrix_l1(A):
  M = np.absolute(A[np.newaxis,:,:]-A[:,np.newaxis,:]).sum(2)
  M = skip_diag_strided(M)
  return M

def distance_matrix_cos(A):
  M = cosine_distances(A)
  M = skip_diag_strided(M)
  return M

def distance_matrix(A):
  M = np.sqrt(np.square(A[np.newaxis,:,:]-A[:,np.newaxis,:]).sum(2))
  M = skip_diag_strided(M)
  return M

def unique_idx(row):
  return np.unique(row, return_index = True)[1]

def idx_unique(a):
  weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
  b = a + weight[:, np.newaxis]
  u, ind = np.unique(b, return_index=True)
  b = np.zeros_like(a)
  np.put(b, ind, np.ones(shape = a.shape).flat[ind])
  return b.astype(bool)

def brute_force_knn_par(A, idx, k, blocksz, edge_mat, w_mat):
  mat = distance_matrix(A)
  dismat = np.hstack((w_mat, mat))
  tile_idx = skip_diag_strided(np.tile(idx, (len(idx), 1)))
  idxmat = np.hstack((edge_mat,  tile_idx ))
  sortdis = np.argsort(dismat, axis = -1)
  dismat = np.sort(dismat, axis = -1)
  idxmat = idxmat[np.arange(dismat.shape[0])[:,None], sortdis]
  unidx = idx_unique(idxmat)
  outdis = np.array([dismat[i, unidx[i]][0:k] for i in range(dismat.shape[0])])
  outidx = np.array([idxmat[i, unidx[i]][0:k] for i in range(dismat.shape[0])])
  return [outidx, outdis]

def brute_force_knn_par_l1(A, idx, k, blocksz, edge_mat, w_mat):
  mat = distance_matrix_l1(A)
  dismat = np.hstack((w_mat, mat))
  tile_idx = skip_diag_strided(np.tile(idx, (len(idx), 1)))
  idxmat = np.hstack((edge_mat,  tile_idx ))
  sortdis = np.argsort(dismat, axis = -1)
  dismat = np.sort(dismat, axis = -1)
  idxmat = idxmat[np.arange(dismat.shape[0])[:,None], sortdis]
  unidx = idx_unique(idxmat)
  outdis = np.array([dismat[i, unidx[i]][0:k] for i in range(dismat.shape[0])])
  outidx = np.array([idxmat[i, unidx[i]][0:k] for i in range(dismat.shape[0])])
  return [outidx, outdis]

def brute_force_knn_par_cos(A, idx, k, blocksz, edge_mat, w_mat):
  mat = distance_matrix_cos(A)
  dismat = np.hstack((w_mat, mat))
  tile_idx = skip_diag_strided(np.tile(idx, (len(idx), 1)))
  idxmat = np.hstack((edge_mat,  tile_idx ))
  sortdis = np.argsort(dismat, axis = -1)
  dismat = np.sort(dismat, axis = -1)
  idxmat = idxmat[np.arange(dismat.shape[0])[:,None], sortdis]
  unidx = idx_unique(idxmat)
  outdis = np.array([dismat[i, unidx[i]][0:k] for i in range(dismat.shape[0])])
  outidx = np.array([idxmat[i, unidx[i]][0:k] for i in range(dismat.shape[0])])
  return [outidx, outdis]

def func_star(a_b):
  try:
    return brute_force_knn_par(*a_b)
  except Exception as e:
    raise Exception(e)

def func_star_l1(a_b):
  try:
    return brute_force_knn_par_l1(*a_b)
  except Exception as e:
    raise Exception(e)

def func_star_cos(a_b):
  try:
    return brute_force_knn_par_cos(*a_b)
  except Exception as e:
    raise Exception(e)
    
def density_broad_search_star(a_b):
  try:
    return euclidean_distances(a_b[1],a_b[0])
  except Exception as e:
    raise Exception(e)

