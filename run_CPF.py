import numpy as np
import pickle
import math
import multiprocessing as mp
from itertools import cycle, islice
import scipy.sparse

from time import time
import csv
import random
from tqdm import tqdm
import gc

from core import CPFcluster
import utils

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings

from sklearn import cluster, datasets, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, KDTree


datasets = ["dermatology", "ecoli", "glass", "letter-recognition", "optdigits", "page-blocks", "pendigits", "Phonemes", "seeds", "vertebral"]
ks = [10, 14, 12, 32, 65, 64, 50, 82, 21, 14]
rhos = [0.4, 0.6, 0.6 ,0.9, 0.9, 0.1, 0.6, 0.5, 0.7, 0.6]
for dataset in datasets:
    Data = np.load("/home/joshuatobin/CPF/July21/Data/Real/" + dataset + ".npy")
    X = Data[:, range(Data.shape[1] - 1)]
    y = Data[:, Data.shape[1] - 1]
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    n, d = X.shape
    X = X.copy(order = 'C')
    for k in ks:
      for rho in rhos:
        t0 = time()
        model = CPFcluster(k = k, rho = rho)
        model.fit(X)
        t1 = time()
        y_pred = model.labels_
        #AMI
        ami = metrics.adjusted_mutual_info_score(y.astype(int), y_pred.astype(int))
        #ARI
        ari = metrics.adjusted_rand_score(y.astype(int), y_pred.astype(int))
        with open("/home/joshuatobin/CPF/July21/Results/CPFkr.csv", 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([dataset,  k, rho, len(np.unique(y_pred)), t1-t0,  ari,ami])

