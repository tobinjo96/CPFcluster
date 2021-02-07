import numpy as np
import CPFcluster
import sys
import getopt
import multiprocessing as np

X = np.genfromtxt('data/synthetic.csv', delimiter = ",")
result = CPFcluster.core.CPFclustering.train(X, k = 5, K = 10, beta = 30, num_hashbits = 12, reps = 30, blocksz = 100, n_core = 20)
np.savetxt('data/synthetic_out.csv', result.y)
print('Labels saves to "data/synthetic.csv".')

