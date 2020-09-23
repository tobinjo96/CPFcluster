import numpy as np
import aghasher
import scipy.sparse
import math
import multiprocessing as mp
import itertools
#import CPFcluster.utils as utils
import utils


class CPFclustering:
  def __init__(self, edges, weights, components, density = np.nan, dists = np.nan, bb = np.nan, y = np.nan):
    self.edges = edges
    self.weights = weights
    self.components = components
    self.density = density
    self.dists = dists
    self.bb = bb
    self.y = y
    
  @classmethod
  def train(cls, X, k, K, beta = 30, reps = 50, num_hashbits = 12, blocksz = 100, n_core = 1):
    if X.shape[0] < 1000:
      edge_arr = np.empty((X.shape[0], k))
      edge_arr[:] = np.nan
      w_arr = np.ones((X.shape[0], k))*np.inf
      PDmat = utils.distance_matrix(X)
      for i in range(len(PDmat)):
        w_arr[i, 0:k] = np.sort(PDmat[i,:])[1:k+1]
        edge_arr[i, 0:k] = np.argsort(PDmat[i,:])[1:k+1]
      
      components = cls.get_components(X, k, edge_arr, w_arr)
      density, dists, bb, edge_arr, w_arr = cls.get_density_dists_bb(X, K, components, reps, num_hashbits, blocksz, n_core)
      y = cls.get_y(edge_arr, w_arr, components, density, dists, bb, beta, k, K)
      result = cls(edge_arr, w_arr, components, density, dists, bb, y)
      return result
    else: 
      edge_arr = np.empty(shape = (X.shape[0], k))
      edge_arr[:] = np.nan
      w_arr = np.ones((X.shape[0], k))*np.inf
      for rep in range(reps):
        edge_arr, w_arr = cls.basic_ann_lsh(X, k, num_hashbits, blocksz, n_core, edge_arr, w_arr)
      
      edge_arr, w_arr = cls.refine(X, edge_arr, w_arr)
      components = cls.get_components(X, k, edge_arr, w_arr)
      density, dists, bb, edge_arr, w_arr = cls.get_density_dists_bb(X, K, components, reps, num_hashbits, blocksz, n_core)
      y = cls.get_y(edge_arr, w_arr, components, density, dists, bb, beta, k, K)
      result = cls(edge_arr, w_arr, components, density, dists, bb, y)      
      return result
  
  @staticmethod
  def basic_ann_lsh(X, k, m, blocksz, n_core, edge_arr, w_arr):
    agh = None
    while agh is None:
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
    edge_arr = edge_arr.astype(int)
    ref_edge_arr = edge_arr
    ref_w_arr = w_arr
    k = edge_arr.shape[1]
    for j in range(k):
      ref_edge_arr = np.hstack((ref_edge_arr, edge_arr[edge_arr[:,j], :]))
      unidx = list(map(utils.unique_idx, ref_edge_arr))
      refined = (np.setdiff1d( ref_edge_arr[row, unidx[row]], row) for row in range(len(unidx)))
      i = 0
      for i_edge_list in refined:
        iw = np.sqrt(np.square(X[i,:] - X[i_edge_list,:]).sum(1))
        ref_w_arr[i,:] = np.sort(iw)[0:k]
        ref_edge_arr[i,0:k] = i_edge_list[np.argsort(iw)[0:k]]
        i += 1
      
      ref_edge_arr = ref_edge_arr[:,0:k]
    
    return ref_edge_arr, ref_w_arr  

  @staticmethod
  def get_components(X, k, edge_arr, w_arr):
    pairs = np.array([ [i ,edge_arr[i,j]] for j in range(k) for i in range(edge_arr.shape[0])])
    dispairs = np.array([ w_arr[i,j] for j in range(k) for i in range(edge_arr.shape[0])])
    n = pairs.max() + 1
    v = pairs[:, 0] + (n * pairs[:, 1])
    # Symmetric values
    v2 = (n * pairs[:, 0]) + pairs[:, 1]
    # Find where symmetric is present
    m = np.isin(v2, v)
    res = pairs[m]
    dispairs = dispairs[m]
    inliers = np.unique(res[:,0]).astype(int)
    CCmat = scipy.sparse.csr_matrix((dispairs, (res[:,0], res[:,1])), shape = (X.shape[0], X.shape[0]))
    CCmatin = CCmat[inliers, :][:, inliers]
    n_components, cc_labels = scipy.sparse.csgraph.connected_components(CCmatin, directed = 'False', return_labels =True)
    components = np.empty(X.shape[0])
    components[:] = np.nan
    components[inliers] = cc_labels.astype(int)
    comp_labs, comp_count = np.unique(components, return_counts = True)
    outlier_components = comp_labs[comp_count < max(3,X.shape[0]/5000)]
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
    for cc in comps:
      cc_idx = [i for i in range(X.shape[0]) if components[i] == cc]
      np_cc_idx = np.array(cc_idx)
      Kcc = min(K, len(components[cc_idx])-1)
      cc_edge_arr = np.empty(shape = (len(cc_idx), Kcc))
      cc_edge_arr[:] = np.nan
      cc_w_arr = np.ones(shape = (len(cc_idx), Kcc))*np.inf
      if len(cc_idx) < 1000:
        PDmat = utils.distance_matrix(X[cc_idx, :])
        cc_w_arr = np.array(list(map(np.sort, PDmat)))[:, 1:Kcc+1]
        cc_edge_arr = np.array(list(map(np.argsort, PDmat))).astype(int)[:,1:Kcc+1]
        density[cc_idx] = np.sum(np.exp(-cc_w_arr), axis = 1)
        cc_density = density[cc_idx]
      else:
        for rep in range(reps):
          cc_edge_arr, cc_w_arr = cls.basic_ann_lsh(X[cc_idx,:], K, num_hashbits, blocksz, n_core, cc_edge_arr, cc_w_arr)
        
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
      for i in cc_search_idx:
        gt_dens_idx = np.where(cc_density > cc_density[i])[0]
        if len(gt_dens_idx) > 0:
          gt_dists = np.sqrt(np.square(X[np.array(cc_idx)[i],:] - X[np.array(cc_idx)[gt_dens_idx],:]).sum(1))
          cc_bb[i] = gt_dens_idx[np.argmin(gt_dists)]
          cc_dists[i] = np.min(gt_dists)
        else:
          cc_bb[i] = i
          cc_dists[i] = np.max(np.sqrt(np.square(X[cc_idx[i],:] - X[cc_idx,:]).sum(1)))
          
      bb[cc_idx] = [cc_idx[i] for i in cc_bb.astype(int)]
      dists[cc_idx] = cc_dists  
    
    return density, dists, bb, edge_arr, w_arr
    

  
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
      for k_c in range(2, min(3*k, len(cc_idx)-2)):
        k_c_edge_arr = cc_edge_arr[:, 0:k_c]
        k_c_w_arr = w_arr[cc_idx, 0:k_c]
        pairs = np.vstack((np.repeat(range(len(cc_idx)), k_c), np.ravel(k_c_edge_arr))).T
        dispairs = np.ravel(k_c_w_arr)
        n = pairs.max() + 1
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
        n = pairs.max() + 1
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
    pairs = np.vstack((np.repeat(inliers, k), np.ravel(k_edge_arr))).T
    dispairs = np.ravel(k_w_arr)
    n = pairs.max() + 1
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
      CompMat = Kmat[cc_idx, :][:,cc_idx]
      if tosplit[cc] is None:
        kconducs[cc] = 0
        kclus[cc] = 1
      elif len(cc_idx) == 0:
        kconducs[cc] = 0
        kclus[cc] = 1
      else:
        cc_gamma = gamma[components == cc]
        for n_clust in range(2, min(beta, len(cc_idx) -1)):
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
  def read_data(fpath, col_idx, metric = np.nan, header = False):
    X = np.genfromtxt(fpath, dtype = str, delimiter = ",", skip_header = int(header))
    if X.shape[1] != len(col_idx):
      sys.exit("Error: length of col_idx must be the same as number of columns of CSV file. ")
    
    if any([x not in [0, 1, 2] for x in col_idx]):
      sys.exit("Error: col_idx entries must be in correct format. ")
    
    del X
    nums = [x for x in range(len(col_idx)) if col_idx[x] == 0 ]
    cats = [x for x in range(len(col_idx)) if col_idx[x] == 1 ]
    X_n = np.genfromtxt(fpath, dtype = float, delimiter = ",", skip_header = int(header))[:, nums]
    for col in range(len(nums)):
      rows = (X_n[:,col] >= np.quantile(X_n[:,col], 0.01)) & (X_n[:,col] <= np.quantile(X_n[:,col], 0.99))
      if np.std(X_n[rows, col]) != 0:
        X_n[:, col] = (X_n[:, col] -np.mean(X_n[rows, col]))/np.std(X_n[rows, col])
      else:
        X_n[:, col] = (X_n[:, col] - np.mean(X_n[:, col]))/np.std(X_n[:, col])
    
    rmcols = []
    X_c = np.genfromtxt(fpath, dtype = str, delimiter = ",", skip_header = int(header))[:, cats]
    for col in range(len(cats)):
      vals, counts = np.unique(X_c[:, col], return_counts = True)
      if len(vals) == 1:
        rmcols.append(col)
    
    cats = [x for x in cats if x not in rmcols]
    del rmcols
    X_c = np.genfromtxt(fpath, dtype = str, delimiter = ",", skip_header = int(header))[:, cats]
    w = [None]*len(cats)
    p = [0]*len(cats)
    l = [None]*len(cats)
    for col in range(len(cats)):
      vals, counts = np.unique(X_c[:, col], return_counts = True)
      l[col] = len(vals)
      if metric is np.nan or 1:
        w[col] = counts/np.sum(counts)
      
      if metric is 2:
        w[col] = np.log(counts/np.sum(counts))/(np.sum(np.log(counts/np.sum(counts))))
      
      for i in range(len(vals) -1):
        for j in range(i+1, len(vals)):
          p[col] += (w[col][i] + w[col][j])*w[col][i]*w[col][j]
      
      p[col] = [1/p[col]]*len(vals)
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
    
    X = np.empty(shape = (X_enc.shape[0], X_n.shape[1] + X_enc.shape[1]))
    nums_t = [len([j for j in range(len(nums)) if nums[j] < nums[i]]) + sum([l[col] for col in range(len(cats)) if cats[col] < nums[i]]) for i in range(len(nums))]
    cats_t = [j for j in range(X.shape[1]) if j not in nums_t]
    X[:, nums_t] = X_n
    del X_n
    X[:, cats_t] = X_enc
    del X_enc
    
    return X

