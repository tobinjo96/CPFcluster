import sys
import CPFcluster
import numpy as np
import argparse
import multiprocessing as mp
from sklearn import metrics 

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


parser = argparse.ArgumentParser(description='Cluster data using the CPFcluster library.')
parser.add_argument('filename',  type=str,
                    help='path to data in .npy format.')
parser.add_argument('outpath', type=str, 
                    help='path to save cluster labels. .csv is recommended.')
parser.add_argument('colidx', type=str,
                    help='list of column indices, where 0 is numeric, 1 is categorical and 2 is label.')
parser.add_argument('--metric', type=int,
                    help='integer to determine weighting scheme for categorical attributes. 1 is flat weighting, 2 is relative frequency weighting and 3 is normalized log weighting. Default value is 1.', 
                    default = 1)
parser.add_argument('--k', type=int,
                    help='number of nearest neighbors used to extract connected components. Default is 5.', 
                    default = 5)
parser.add_argument('--K', type=int,
                    help='number of nearest neighbors used to compute density. Default is 20.', 
                    default = 20)
parser.add_argument('--beta', type=int,
                    help='maximum number of candidate centers assessed for each compoment. Default is 30. ', 
                    default = 30)
parser.add_argument('--reps', type=int,
                    help='Number of repetitions of the locality sensitive hashing method used in computing the nearest-neighbor graphs. Default is 30.', 
                    default = 30)
parser.add_argument('--num_hashbits', type=int,
                    help='Number of hashbits used in locality sensitive hashing method. Default is 12.', 
                    default = 12)
parser.add_argument('--blocksz', type=int,
                    help='Size of the neighborhood on which brute force kNN is computed in locality sensitive hashing method. Default is 100', 
                    default = 100)
parser.add_argument('--n_core', type=int,
                    help='Number of processors to be used when computing nearest-neighbor graph. If set to 1, parallel processing does not take place. Default is all available cores.', 
                    default = mp.cpu_count())


def set_args():
  args = parser.parse_args()
  
  for opt, val in vars(args).items():
    if opt == 'filename':
      filename = str(val)
    elif opt == 'outpath':
      outpath = val
    elif opt == 'colidx':
      colidx = val
    elif opt == 'metric':
      metric = val
    elif opt == 'k':
      k = val
    elif opt == 'K':
      K = val
    elif opt == 'beta':
      beta = val
    elif opt == 'reps':
      reps = val
    elif opt == 'num_hashbits':
      num_hashbits = val
    elif opt == 'blocksz':
      blocksz = val
    elif opt == 'n_core':
      n_core = val
  
  return [filename, outpath, colidx, metric, k, K, beta, reps, num_hashbits, blocksz, n_core]


filename, outpath, colidx, metric, k, K, beta, reps, num_hashbits, blocksz, n_core = set_args()
outname = (filename)[0:len(filename) -4] + "_y" + "k" + str(k) + "K" + str(K) + ".npy"

print(colidx)
X = CPFcluster.CPFclustering.prep_data(filename, colidx, metric)
if 2 in colidx:
  y = X[:, X.shape[1]-1]
  X = X[:, range(X.shape[1]-1)]
  result = CPFcluster.CPFclustering.train(X, k, K, beta, reps, num_hashbits, blocksz, n_core)
  #Purity
  pur = purity_score(y.astype(int), result.y.astype(int))
  #AMI
  ami = metrics.adjusted_mutual_info_score(y[inidx].astype(int), result.y[inidx].astype(int))
  #ARI
  ari = metrics.adjusted_rand_score(y[inidx].astype(int), result.y[inidx].astype(int))
  np.savetxt(result.y, outpath)
  print("Purity Score: "+ pur + " ARI: " + ari + " AMI: " + ami + " for " + filename + " with k = " + k + " and K = " + K + ". Labels saved to " + outpath +"." )
else:
  result = CPFcluster.CPFclustering.train(X, k, K, beta, reps, num_hashbits, blocksz, n_core)
  np.savetxt(result.y, outpath)
  print("Labels saved to " + outpath +"." )

  



