import numpy as np
from CPFcluster import CPFcluster
import warnings

from sklearn import cluster, datasets, metrics
import csv

Data = np.load("Dermatology.npy")
X = Data[:, range(X.shape[1] - 1)]
y = Data[:, X.shape[1] - 1]
# normalize dataset for easier parameter selection
model = CPFcluster(k = 13, rho = 0.3, alpha = 1)
model.fit(X)
ami = metrics.adjusted_mutual_info_score(y.astype(int), model.memberships.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int),  model.memberships.astype(int))
with open("CPF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Dermatology",  k, rho, len(np.unique(model.memberships)), ari,ami])

Data = np.load("Ecoli.npy")
X = Data[:, range(X.shape[1] - 1)]
y = Data[:, X.shape[1] - 1]
# normalize dataset for easier parameter selection
model = CPFcluster(k = 13, rho = 0.6, alpha = 1)
model.fit(X)
ami = metrics.adjusted_mutual_info_score(y.astype(int), model.memberships.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int),  model.memberships.astype(int))
with open("CPF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Ecoli",  k, rho, len(np.unique(model.memberships)), ari,ami])

Data = np.load("Glass.npy")
X = Data[:, range(X.shape[1] - 1)]
y = Data[:, X.shape[1] - 1]
# normalize dataset for easier parameter selection
model = CPFcluster(k = 13, rho = 0.5, alpha = 1)
model.fit(X)
ami = metrics.adjusted_mutual_info_score(y.astype(int), model.memberships.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int),  model.memberships.astype(int))
with open("CPF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Glass",  k, rho, len(np.unique(model.memberships)), ari,ami])

Data = np.load("Letter-Recognition.npy")
X = Data[:, range(X.shape[1] - 1)]
y = Data[:, X.shape[1] - 1]
# normalize dataset for easier parameter selection
model = CPFcluster(k = 25, rho = 0.9, alpha = 1)
model.fit(X)
ami = metrics.adjusted_mutual_info_score(y.astype(int), model.memberships.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int),  model.memberships.astype(int))
with open("CPF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Letter-Recognition",  k, rho, len(np.unique(model.memberships)), ari,ami])
