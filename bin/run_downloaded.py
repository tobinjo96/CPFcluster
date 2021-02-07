import numpy as np
import CPFcluster.core as core
import pickle
import math
from sklearn import metrics
from time import time
import csv
import random
from tqdm import tqdm
import gc
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

#Running CPF for the attached sample of the KDD Cup '99 data set.
k = 50
K = 20
dataset = "kddcup_1"
fpath ="data/" + dataset + ".npy"
X = np.load(fpath)
y = X[:, X.shape[1]-1]
X = X[:, range(X.shape[1]-1)]
try:
  t1 = time()
  result = core.CPFclustering.train(X, k, K, beta = 30, reps = 30, num_hashbits = 12, blocksz = 100, n_core = 20)
  t2 = time()
  inidx = ~np.isnan(result.y)
  #Purity
  pur = purity_score(y.astype(int), result.y.astype(int))
  #AMI
  ami = metrics.adjusted_mutual_info_score(y[inidx].astype(int), result.y[inidx].astype(int))
  #ARI
  ari = metrics.adjusted_rand_score(y[inidx].astype(int), result.y[inidx].astype(int))
  with open("out.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow([dataset,  k, K, t2-t1, ari,pur, ami])
except Exception as e:
  with open("errors.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow([dataset, k, K, e])

#Running CPF for the attached sample of the YTB-Faces data set.
k = 10
K = 20
dataset = "YTB"
fpath = "data/" + dataset + ".npy"
X = np.load(fpath)
y = X[:, X.shape[1]-1]
X = X[:, range(X.shape[1]-1)]
try:
  t1 = time()
  result = core.CPFclustering.train(X, k, K, beta = 30, reps = 30, num_hashbits = 12, blocksz = 100, n_core = 20)
  t2 = time()
  inidx = ~np.isnan(result.y)
  #Purity
  pur = purity_score(y.astype(int), result.y.astype(int))
  #AMI
  ami = metrics.adjusted_mutual_info_score(y[inidx].astype(int), result.y[inidx].astype(int))
  #ARI
  ari = metrics.adjusted_rand_score(y[inidx].astype(int), result.y[inidx].astype(int))
  with open("out.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow([dataset,  k, K, t2-t1, ari,pur, ami])
except Exception as e:
  with open("errors.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow([dataset, k, K, e])
    
#Running CPF for the attached sample of the MSCM1 data set.     
k = 5
K = 20
dataset = "MSCM1"
fpath = "data/" +dataset + ".npy"
X = np.load(fpath)
y = X[:, X.shape[1]-1]
X = X[:, range(X.shape[1]-1)]
try:
  t1 = time()
  result = core.CPFclustering.train(X, k, K, beta = 30, reps = 30, num_hashbits = 12, blocksz = 100, n_core = 20)
  t2 = time()
  inidx = ~np.isnan(result.y)
  #Purity
  pur = purity_score(y.astype(int), result.y.astype(int))
  #AMI
  ami = metrics.adjusted_mutual_info_score(y[inidx].astype(int), result.y[inidx].astype(int))
  #ARI
  ari = metrics.adjusted_rand_score(y[inidx].astype(int), result.y[inidx].astype(int))
  with open("out.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow([dataset,  k, K, t2-t1, ari,pur, ami])
except Exception as e:
  with open("errors.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow([dataset, k, K, e])


