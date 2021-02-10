import numpy as np
import sys
import aghasher
import CPFcluster.aghashercore as aghashercore
import scipy.sparse
import math
import multiprocessing as mp
import itertools
import pandas as pd
import CPFcluster.utils as utils
import gc
from tqdm import tqdm

class CPFclustering:
  def __init__(self, edges, weights, components, density = np.nan, dists = np.nan, bb = np.nan, y = np.nan, ps = np.nan):
    self.edges = edges
    self.weights = weights
    self.components = components
    self.density = density
    self.dists = dists
    self.bb = bb
    self.y = y
    self.ps = ps
    
  @classmethod
  def train(cls, X, k, K, beta = 30, reps = 50, num_hashbits = 12, blocksz = 100, n_core = 1):
    if X.shape[0] < 1000:
      edge_arr = np.empty((X.shape[0], k))
      edge_arr[:] = np.nan
      w_arr = np.ones((X.shape[0], k))*np.inf
      PDmat = utils.distance_matrix(X)
      for i in range(len(PDmat)):
        w_arr[i, 0:k] = np.sort(PDmat[i,:])[range(k)]
        edge_arr[i, 0:k] = np.argsort(PDmat[i,:])[range(k)]
      
      components = cls.get_components(X, k, edge_arr, w_arr)
      density, dists, bb, edge_arr, w_arr, ps = cls.get_density_dists_bb(X, K, components, reps, num_hashbits, blocksz, n_core)
      y = cls.get_y(edge_arr, w_arr, components, density, dists, bb, beta, k, K)
      result = cls(edge_arr, w_arr, components, density, dists, bb, y, ps)
      return result
    else: 
      edge_arr = np.empty(shape = (X.shape[0], k))
      edge_arr[:] = np.nan
      w_arr = np.ones((X.shape[0], k))*np.inf
      for rep in tqdm(range(reps), "Data ANN "):
        edge_arr, w_arr = cls.basic_ann_lsh(X, k, num_hashbits, blocksz, n_core, edge_arr, w_arr)
      
      edge_arr, w_arr = cls.refine(X, edge_arr, w_arr)
      components = cls.get_components(X, k, edge_arr, w_arr)
      del edge_arr, w_arr
      density, dists, bb, edge_arr, w_arr, ps = cls.get_density_dists_bb(X, K, components, reps, num_hashbits, blocksz, n_core)
      del X
      y = cls.get_y(edge_arr, w_arr, components, density, dists, bb, beta, k, K)
      result = cls(edge_arr, w_arr, components, density, dists, bb, y, ps)      
      return result
  
  @staticmethod
  def basic_ann_lsh(X, k, m, blocksz, n_core, edge_arr, w_arr):
    agh = None
    while agh is None:
      try:
        anchoridx = np.random.choice(X.shape[0], size = 25, replace = False)
        anchor = X[anchoridx, :]
        agh, H_train = aghashercore.AnchorGraphHasher.train(X, anchor, m)
      except:
        try:
          anchoridx = np.random.choice(X.shape[0], size = 25, replace = False)
          anchor = X[anchoridx, :]
          agh, H_train = aghasher.AnchorGraphHasher.train(X, anchor, m)
        except: 
          pass
    
    Y = agh.hash(X)
    Y = Y.astype(int)
    w = np.random.rand(m, 1)
    p = Y.dot(w)
    p = np.ravel(p)
    pid = np.argsort(p)
    l = ([X[block,: ], block, k, blocksz, edge_arr[block,:], w_arr[block,:]] for block in list(utils.chunks(pid, blocksz)))
    try:
      pool = mp.Pool(processes=n_core)              
      #result = [[]]*math.ceil(len(list(utils.chunks(pid, blocksz)))/n_core)
      result = []
      N = max(math.floor(X.shape[0]/(100*blocksz)), 20)
      while True:
        g2 = pool.map(utils.func_star, itertools.islice(l, N))
        if g2:
          result.append(g2)
        else:
          break
      pool.terminate()
    except Exception as e:
      print("POOL ERROR")
      pool.close()
      pool.terminate()
    
    res_flat = [res_pair for res_list in result for res_pair in res_list]
    repedge_arr, repw_arr = zip(*res_flat)
    repedge_arr = np.vstack(repedge_arr)
    repw_arr = np.vstack(repw_arr)
    repedge_arr = repedge_arr[np.argsort(pid),:]
    repw_arr = repw_arr[np.argsort(pid),:]
    return repedge_arr, repw_arr
  
  @staticmethod
  def refine(X, edge_arr, w_arr):
    n = edge_arr.shape[0]
    k = edge_arr.shape[1]
    edge_arr = edge_arr.astype(int)
    in_edge_arr = edge_arr.astype(int)
    chunks = utils.chunks(range(n), 20000)
    for j in range(k):
      for chunk in chunks:
        ref_edge_arr = in_edge_arr[in_edge_arr[chunk,j], :]
        stack_edge_arr = np.hstack((edge_arr[chunk,:], ref_edge_arr))
        X_NNchunk = X[ref_edge_arr, :]
        ref_w_arr = np.sqrt(np.square(X[chunk,np.newaxis,:] - X_NNchunk).sum(2))
        stack_w_arr = np.hstack((w_arr[chunk,:], ref_w_arr))
        unidx = [utils.unique_idx(stack_edge_arr[i,:]) for i in range(len(chunk))]
        unidx = [unidx[i][stack_edge_arr[i, unidx[i]] != chunk[i]] for i in range(len(chunk))]
        sortidx = [np.argsort(stack_w_arr[i, unidx[i]])[range(k)] for i in range(len(chunk))]
        edge_arr[chunk,:] = np.stack([stack_edge_arr[i, unidx[i]][sortidx[i]] for i in range(len(chunk))])
        w_arr[chunk,:] = np.stack([stack_w_arr[i, unidx[i]][sortidx[i]] for i in range(len(chunk))])
    
    return edge_arr, w_arr
  
  @staticmethod
  def get_components(X, k, edge_arr, w_arr):
    w_arr = w_arr.ravel()
    edge_arr = np.hstack((np.repeat(range(edge_arr.shape[0]), k)[:,np.newaxis], edge_arr.ravel()[:,np.newaxis]))
    n = edge_arr[~np.isnan(edge_arr)].max() + 1
    v = edge_arr[:, 0] + (n * edge_arr[:, 1])
    # Symmetric values
    v2 = (n * edge_arr[:, 0]) + edge_arr[:, 1]
    # Find where symmetric is present
    m = np.zeros((v2.shape[0]), dtype = bool)
    for chunk in utils.chunks(range(v2.shape[0]), 200000):
      m[chunk] = np.isin(v2[chunk], v)
    del v, v2
    gc.collect()
    res = edge_arr[m]
    del edge_arr 
    gc.collect()
    w_arr = w_arr[m]
    inliers = np.unique(res[:,0]).astype(int)
    CCmat = scipy.sparse.csr_matrix((w_arr, (res[:,0], res[:,1])), shape = (X.shape[0], X.shape[0]))
    del w_arr
    CCmatin = CCmat[inliers, :][:, inliers]
    n_components, cc_labels = scipy.sparse.csgraph.connected_components(CCmatin, directed = 'False', return_labels =True)
    components = np.empty(X.shape[0])
    components[:] = np.nan
    components[inliers] = cc_labels.astype(int)
    comp_labs, comp_count = np.unique(components, return_counts = True)
    outlier_components = comp_labs[comp_count < 25]
    nanidx = np.where(np.in1d(components, outlier_components))
    components[nanidx] = np.nan
    
    return components

  @staticmethod
  def get_components_img(X, k, edge_arr, w_arr):
    w_arr = w_arr.ravel()
    edge_arr = np.hstack((np.repeat(range(edge_arr.shape[0]), k)[:,np.newaxis], edge_arr.ravel()[:,np.newaxis]))
    n = edge_arr[~np.isnan(edge_arr)].max() + 1
    v = edge_arr[:, 0] + (n * edge_arr[:, 1])
    # Symmetric values
    v2 = (n * edge_arr[:, 0]) + edge_arr[:, 1]
    # Find where symmetric is present
    m = np.zeros((v2.shape[0]), dtype = bool)
    for chunk in utils.chunks(range(v2.shape[0]), 200000):
      m[chunk] = np.isin(v2[chunk], v)
    del v, v2
    gc.collect()
    res = edge_arr[m]
    del edge_arr 
    gc.collect()
    w_arr = w_arr[m]
    inliers = np.unique(res[:,0]).astype(int)
    CCmat = scipy.sparse.csr_matrix((w_arr, (res[:,0], res[:,1])), shape = (X.shape[0], X.shape[0]))
    del w_arr
    CCmatin = CCmat[inliers, :][:, inliers]
    n_components, cc_labels = scipy.sparse.csgraph.connected_components(CCmatin, directed = 'False', return_labels =True)
    components = np.empty(X.shape[0])
    components[:] = np.nan
    components[inliers] = cc_labels.astype(int)
    comp_labs, comp_count = np.unique(components, return_counts = True)
    outlier_components = comp_labs[comp_count < 10]
    nanidx = np.where(np.in1d(components, outlier_components))
    components[nanidx] = np.nan
    
    return components
  
  @classmethod
  def get_density_dists_bb(cls, X, K, components, reps = 50, num_hashbits = 12, blocksz = 100, n_core = 1):
    density = np.empty((X.shape[0]))
    density[:] = np.nan
    dists = np.empty((X.shape[0]))
    dists[:] = np.nan
    bb = np.empty((X.shape[0]))
    bb[:] = np.nan
    comps = np.unique((components[~np.isnan(components)])).astype(int)
    edge_arr = np.empty(shape = (X.shape[0], K))
    edge_arr[:] = np.nan
    w_arr = np.empty(shape = (X.shape[0], K))
    w_arr[:] = np.nan
    ps = np.zeros((1, 2))
    for cc in comps:
      print(cc)
      cc_idx = [i for i in range(X.shape[0]) if components[i] == cc]
      np_cc_idx = np.array(cc_idx)
      Kcc = min(K, len(components[cc_idx])-1)
      cc_edge_arr = np.empty(shape = (len(cc_idx), Kcc))
      cc_edge_arr[:] = np.nan
      cc_w_arr = np.ones(shape = (len(cc_idx), Kcc))*np.inf
      if len(cc_idx) < 1000:
        PDmat = utils.distance_matrix(X[cc_idx, :])
        cc_w_arr = np.array(list(map(np.sort, PDmat)))[:, range(Kcc)]
        cc_edge_arr = np.array(list(map(np.argsort, PDmat))).astype(int)[:,range(Kcc)]
        density[cc_idx] = np.sum(np.exp(-cc_w_arr), axis = 1)
        cc_density = density[cc_idx]
      else:
        nrep = 25
        for rep in tqdm(range(nrep), "Component ANN"):
          cc_edge_arr, cc_w_arr = cls.basic_ann_lsh(X[cc_idx,:], K, num_hashbits, blocksz, n_core, cc_edge_arr, cc_w_arr)
        
        narows = [x for x in range(cc_edge_arr.shape[0]) if np.inf in cc_edge_arr[x, :]]
        for x in narows:
          na_dists = np.sqrt(np.square(X[np.array(cc_idx)[x],:] - X[np.array(cc_idx),:]).sum(1))
          cc_w_arr[x,:] = np.sort(na_dists)[range(1, cc_edge_arr.shape[1] + 1)]
          cc_edge_arr[x, :] = np.argsort(na_dists)[range(1, cc_edge_arr.shape[1] + 1)]
        
        cc_edge_arr, cc_w_arr = cls.refine(X[cc_idx,:], cc_edge_arr, cc_w_arr)
        density[cc_idx] = np.sum(np.exp(-cc_w_arr), axis = 1)
        cc_density = density[cc_idx]
      
      edge_arr[cc_idx,0:Kcc] = np_cc_idx[cc_edge_arr]
      w_arr[cc_idx,0:Kcc] = cc_w_arr
      cc_bb = np.empty((len(cc_idx)))
      cc_bb[:] = np.nan
      cc_dists = np.empty((len(cc_idx)))
      cc_dists[:] = np.nan
      cc_dens_diff = cc_density[:, np.newaxis] - cc_density[cc_edge_arr]
      rows, cols = np.where(cc_dens_diff < 0)
      rows, unidx = np.unique(rows, return_index =  True)
      cols = cols[unidx]
      cc_bb[rows] = cc_edge_arr[rows, cols]
      cc_dists[rows] = cc_w_arr[rows, cols]
      cc_search_idx = list(np.setdiff1d(list(range(len(cc_idx))), rows))
      ps = np.vstack((ps, [len(cc_idx), len(cc_search_idx)/len(cc_idx)]))
      for cc_indx_chunk in tqdm(utils.chunks(cc_search_idx, 100), "Component Broad Search"):
        cc_search_dens = cc_density[cc_indx_chunk]
        a =  cc_density > cc_search_dens[:, np.newaxis] 
        if any(np.sum(a, axis = 1) == 0):
          max_i = [i for i in range(a.shape[0]) if np.sum(a[i,:]) ==0]
          if len(max_i) > 1:
            for max_j in max_i[1:len(max_i)]:
              a[max_j, cc_indx_chunk[max_i[0]]] = True
          max_i = max_i[0]
          cc_bb[cc_indx_chunk[max_i]] = cc_indx_chunk[max_i]
          cc_dists[cc_indx_chunk[max_i]] = np.max(np.sqrt(np.square(X[cc_idx[cc_indx_chunk[max_i]],:] - X[cc_idx,:]).sum(1)))
          del cc_indx_chunk[max_i]
          a = np.delete(a, max_i, 0)
        
        gt_dists = ([X[np.array(cc_idx)[cc_indx_chunk[i]],np.newaxis], X[np.array(cc_idx)[a[i,:]],:]] for i in range(len(cc_indx_chunk)))
        if (a.shape[0]>5):
          try:
            pool = mp.Pool(processes=n_core)              
            argmin_cc = []
            N = 5
            while True:
              g2 = pool.map(utils.density_broad_search_star, itertools.islice(gt_dists, N))
              if g2:
                argmin_cc.append(g2)
              else:
                break
            pool.terminate()
          except Exception as e:
            print("POOL ERROR: "+ e)
            pool.close()
            pool.terminate()
        else:
          argmin_cc = list(map(utils.density_broad_search_star, list(gt_dists)))
        dis_flat = [dis_pair for dis_list in argmin_cc for dis_pair in dis_list]
        argmins = [np.argmin(l) for l in dis_flat]
        for i in range(a.shape[0]):
          cc_bb[cc_indx_chunk[i]] = np.where(a[i,:] == 1)[0][argmins[i]]
          cc_dists[cc_indx_chunk[i]] = dis_flat[i][argmins[i]]
      
      bb[cc_idx] = [cc_idx[i] for i in cc_bb.astype(int)]
      dists[cc_idx] = cc_dists  
    
    return density, dists, bb, edge_arr, w_arr, ps
  
  @staticmethod
  def get_y(edge_arr, w_arr, components, density, dists, bb, beta, k, K):
    comps = np.unique((components[~np.isnan(components)])).astype(int)
    gamma = density * dists
    tosplit = [None]*(max(comps)+1)
    kmin = [None]*(max(comps)+1)
    nx = components.shape[0]
    for cc in comps:
      conducs = [0, 0]
      cc_idx = np.array([i for i in range(nx) if components[i] == cc])
      cc_key = np.empty(nx)
      cc_key[cc_idx] = range(len(cc_idx))
      cc_edge_arr = edge_arr[cc_idx,  0:len(cc_idx)-1].astype(int)
      cc_edge_arr = cc_key[cc_edge_arr].astype(int)
      cc_gamma = gamma[components == cc]
      centrTF = np.zeros((cc_idx.shape[0]), dtype = bool)
      centrTF[np.argsort(cc_gamma, axis = 0)[::-1][range(2)]] = True
      BBTree = np.zeros((cc_idx.shape[0], 2))
      BBTree[:, 0] = cc_idx
      BBTree[:, 1] = bb[cc_idx]
      index = np.argsort(BBTree[:, 0])
      sorted_x = BBTree[index, 0]
      sorted_index = np.searchsorted(sorted_x, BBTree[:, 1])
      yindex = np.take(index, sorted_index, mode="clip")
      BBTree[:,0] = index.astype(int)
      BBTree[:, 1] = yindex.astype(int)
      BBTree[centrTF,1] = BBTree[centrTF,0]
      BBTree = np.array(BBTree, dtype = int)
      Clustmat = scipy.sparse.csr_matrix((np.ones((cc_idx.shape[0])), (BBTree[:,0], BBTree[:, 1])), shape = (cc_idx.shape[0], cc_idx.shape[0]))
      n_clusts, assignlab = scipy.sparse.csgraph.connected_components(Clustmat, directed = 'True', return_labels =True)
      for k_c in range(2, min(3*k, cc_edge_arr.shape[1]-2)):
        k_c_edge_arr = cc_edge_arr[:, 0:k_c]
        k_c_w_arr = w_arr[cc_idx, 0:k_c]
        pairs = np.vstack((np.repeat(range(len(cc_idx)), k_c), np.ravel(k_c_edge_arr))).T
        dispairs = np.ravel(k_c_w_arr)
        n = pairs[~np.isnan(pairs)].max() + 1
        v = pairs[:, 0] + (n * pairs[:, 1])
        v2 = (n * pairs[:, 0]) + pairs[:, 1]
        m = np.isin(v2, v)
        res = pairs[m]
        dispairs = dispairs[m]
        Kmat = scipy.sparse.csr_matrix((dispairs, (res[:,0], res[:,1])), shape = (len(cc_idx), len(cc_idx)))
        if scipy.sparse.csgraph.connected_components(Kmat[assignlab == 0,:][:, assignlab ==0], directed = True)[0] == 1 and scipy.sparse.csgraph.connected_components(Kmat[assignlab == 1,:][:, assignlab ==1], directed = True)[0] == 1 :
          kmin[cc] = k_c
          conducs[0] = np.sum(np.exp(-Kmat[assignlab == 0,:][:,assignlab == 1].data))/min(np.sum(np.exp(-Kmat[assignlab == 0,:][:,assignlab == 0].data)), np.sum(np.exp(-Kmat[assignlab == 1,:][:,assignlab == 1].data)))
          break
      
      if kmin[cc] is not None:
        k_c += 1
        k_c_edge_arr = cc_edge_arr[:, 0:k_c]
        k_c_w_arr = w_arr[cc_idx, 0:k_c]
        pairs = np.vstack((np.repeat(range(len(cc_idx)), k_c), np.ravel(k_c_edge_arr))).T
        dispairs = np.ravel(k_c_w_arr)
        n = pairs[~np.isnan(pairs)].max() + 1
        v = pairs[:, 0] + (n * pairs[:, 1])
        v2 = (n * pairs[:, 0]) + pairs[:, 1]
        m = np.isin(v2, v)
        res = pairs[m]
        dispairs = dispairs[m]    
        Kmat = scipy.sparse.csr_matrix((dispairs, (res[:,0], res[:,1])), shape = (len(cc_idx), len(cc_idx)))
        conducs[1] = np.sum(np.exp(-Kmat[assignlab == 0,:][:,assignlab == 1].data))/min(np.sum(np.exp(-Kmat[assignlab == 0,:][:,assignlab == 0].data)), np.sum(np.exp(-Kmat[assignlab == 1,:][:,assignlab == 1].data)))
        if conducs[0] > conducs[1]:
          tosplit[cc] = cc
        
    
    inliers = [x for x in range(nx) if not np.isnan(components[x])]
    k_edge_arr = edge_arr[inliers, 0:k]
    k_w_arr = w_arr[inliers, 0:k]
    pairs = np.vstack((np.repeat(inliers, k_edge_arr.shape[1]), np.ravel(k_edge_arr))).T
    dispairs = np.ravel(k_w_arr)
    n = pairs[~np.isnan(pairs)].max() + 1
    v = pairs[:, 0] + (n * pairs[:, 1])
    v2 = (n * pairs[:, 0]) + pairs[:, 1]
    m = np.isin(v2, v)
    res = pairs[m]
    dispairs = dispairs[m]    
    Kmat = scipy.sparse.csr_matrix((dispairs, (res[:,0], res[:,1])), shape = (nx, nx))
    kconducs = [float('Inf')]*(max(comps)+1)
    kclus = [0]*(max(comps)+1)
    maxcc = 0
    for cc in comps:
      cc_idx = np.array([i for i in range(nx) if components[i] == cc]).astype(int)
      cc_edge_arr = edge_arr[cc_idx,  0:len(cc_idx)-1].astype(int)
      cc_edge_arr = cc_key[cc_edge_arr].astype(int)
      CompMat = Kmat[cc_idx, :][:,cc_idx]
      if tosplit[cc] is None:
        kconducs[cc] = 0
        kclus[cc] = 1
      elif len(cc_idx) == 0:
        kconducs[cc] = 0
        kclus[cc] = 1
      else:
        cc_gamma = gamma[components == cc]
        print(cc)
        for n_clust in range(2, min(beta, min(len(cc_idx), cc_edge_arr.shape[1]) -1)):
          maxcc = 0
          centrTF = np.zeros((cc_idx.shape[0]), dtype = bool)
          centrTF[np.argsort(cc_gamma, axis = 0)[::-1][range(n_clust)]] = True
          BBTree = np.zeros((cc_idx.shape[0], 2))
          BBTree[:, 0] = cc_idx
          BBTree[:, 1] = bb[cc_idx]
          index = np.argsort(BBTree[:, 0])
          sorted_x = BBTree[index, 0]
          sorted_index = np.searchsorted(sorted_x, BBTree[:, 1])
          yindex = np.take(index, sorted_index, mode="clip")
          BBTree[:,0] = index.astype(int)
          BBTree[:, 1] = yindex.astype(int)
          BBTree[centrTF,1] = BBTree[centrTF,0]
          BBTree = np.array(BBTree, dtype = int)
          Clustmat = scipy.sparse.csr_matrix((np.ones((cc_idx.shape[0])), (BBTree[:,0], BBTree[:, 1])), shape = (cc_idx.shape[0], cc_idx.shape[0]))
          n_clusts, assignlab = scipy.sparse.csgraph.connected_components(Clustmat, directed = 'True', return_labels =True)
          conducc = []
          for cl in range(n_clust):
              conducc.append(np.sum(np.exp(-CompMat[assignlab == cl,:][:,assignlab != cl].data))/min(np.sum(np.exp(-CompMat[assignlab == cl,:][:,assignlab == cl].data)), np.sum(np.exp(-CompMat[assignlab != cl,:][:,assignlab != cl].data))))
              if max(conducc) > maxcc:
                maxcc = max(conducc)
          
          if maxcc < kconducs[cc]:
            kconducs[cc] = maxcc
            kclus[cc] = n_clust
    
    clplus = 0
    cllabels = [None]*(max(comps)+1)
    y = np.empty(nx)
    y[:] = np.nan
    for cc in comps:
      cc_idx = np.array([i for i in range(nx) if components[i] == cc]).astype(int)
      cc_gamma = gamma[components == cc]
      centrTF = np.zeros((cc_idx.shape[0]), dtype = bool)
      centrTF[np.argsort(cc_gamma, axis = 0)[::-1][range(kclus[cc])]] = True
      BBTree = np.zeros((cc_idx.shape[0], 2))
      BBTree[:, 0] = cc_idx
      BBTree[:, 1] = bb[cc_idx]
      index = np.argsort(BBTree[:, 0])
      sorted_x = BBTree[index, 0]
      sorted_index = np.searchsorted(sorted_x, BBTree[:, 1])
      yindex = np.take(index, sorted_index, mode="clip")
      BBTree[:,0] = index.astype(int)
      BBTree[:, 1] = yindex.astype(int)
      BBTree[centrTF,1] = BBTree[centrTF,0]
      BBTree = np.array(BBTree, dtype = int)
      Clustmat = scipy.sparse.csr_matrix((np.ones((cc_idx.shape[0])), (BBTree[:,0], BBTree[:, 1])), shape = (cc_idx.shape[0], cc_idx.shape[0]))
      n_clusts, assignlab = scipy.sparse.csgraph.connected_components(Clustmat, directed = 'True', return_labels =True)
      assignlab = assignlab + clplus
      cllabels[cc] = assignlab
      clplus += n_clusts
      y[cc_idx] = cllabels[cc]
    
    return y
  
  @staticmethod 
  def to_npy(inpath, outpath, header = False):
    df = []
    chunksize = 10 ** 6
    for chunk in pd.read_csv(inpath, chunksize=chunksize):
      df.append(chunk.values)    
    
    X = np.zeros((chunksize*(len(df)-1) + df[len(df)-1].shape[0],df[0].shape[1]))
    for i in range(len(df)):
      X[(i*df[i-1].shape[0]):(i*df[0].shape[0] + df[i].shape[0])] = df[i]
    
    np.save(X, outpath)
  
  @staticmethod
  def prep_data(fpath, colidx, metric = 1):
    X = np.load(fpath)
    if X.shape[1] != len(colidx):
      sys.exit("Error: length of colidx must be the same as number of columns of NPY array. ")
    
    if any([x not in [0, 1, 2] for x in colidx]):
      sys.exit("Error: colidx entries must be in correct format. ")
    
    nums = [x for x in range(len(colidx)) if colidx[x] == 0 ]
    cats = [x for x in range(len(colidx)) if colidx[x] == 1 ]
    ys = [x for x in range(len(colidx)) if colidx[x] == 2]
    X_n = X[:, nums]
    for col in range(len(nums)):
      rows = (X[:,nums[col]] >= np.quantile(X[:,nums[col]], 0.01)) & (X[:,nums[col]] <= np.quantile(X[:,nums[col]], 0.99))
      if np.std(X[rows, nums[col]]) != 0:
        X[:, nums[col]] = (X[:, nums[col]] -np.mean(X[rows, nums[col]]))/np.std(X[rows, nums[col]])
      else:
        X[:, nums[col]] = (X[:, nums[col]] - np.mean(X[:, nums[col]]))/np.std(X[:, nums[col]])
    
    rmcols = []
    X_c = X[:, cats]
    for col in range(len(cats)):
      vals, counts = np.unique(X_c[:, col], return_counts = True)
      if len(vals) == 1:
        rmcols.append(col)
    
    cats = [x for x in cats if x not in rmcols]
    del rmcols
    X_c = X[:, cats]
    w = [None]*len(cats)
    p = [0]*len(cats)
    l = [None]*len(cats)
    for col in range(len(cats)):
      vals, counts = np.unique(X[:, cats[col]], return_counts = True)
      l[col] = len(vals)
      if metric == 1:
        w[col] = [1]*len(vals)
      
      if metric == 2:
        w[col] = counts/np.sum(counts)
      
      if metric == 3:
        w[col] = np.log(counts/np.sum(counts))/(np.sum(np.log(counts/np.sum(counts))))
      
      for i in range(len(vals)):
        for j in range(len(vals)):
          if j == i:
            continue
          p[col] += (w[col][i] + w[col][j])*(counts[i]/np.sum(counts))*(counts[j]/np.sum(counts))
      
      p[col] = np.sqrt(2/p[col])
      w[col] = np.sqrt(w[col])
      dmy = np.searchsorted(vals, X_c[:, col])
      shape = (dmy.size, dmy.max()+1)
      one_hot = np.zeros(shape)
      rows = np.arange(dmy.size)
      one_hot[rows, dmy] = 1
      one_hot = one_hot.astype(int)
      one_hot = one_hot*w[col]*p[col]
      if col == 0:
        X_enc = one_hot
      else:
        X_enc = np.hstack((X_enc, one_hot))
    
    X_out = np.empty(shape = (X_enc.shape[0], len(nums)+ X_enc.shape[1]))
    nums_t = [len([j for j in range(len(nums)) if nums[j] < nums[i]]) + sum([l[col] for col in range(len(cats)) if cats[col] < nums[i]]) for i in range(len(nums))]
    cats_t = [j for j in range(X_out.shape[1]) if j not in nums_t]
    X_out[:, nums_t] = X[:, nums]
    #del X
    X_out[:, cats_t] = X_enc
    del X_enc
    if len(ys) == 1:
      X_out = np.hstack((X_out, X[:, ys]))
    
    return X_out
  
  
  
  
