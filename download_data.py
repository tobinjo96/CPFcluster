import numpy as np
import unlzw3
from pathlib import Path
import urllib.request as urllib
import csv 

#Dermatology
Data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data', dtype = float, delimiter = ",")
np.save("Dermatology.npy", Data)

#Ecoli
Data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data', dtype = str)
X = Data[:, range(1, Data.shape[1]-1)].astype(float)
y = np.unique(Data[:, Data.shape[1]-1], return_inverse=True)[1]
Data = np.hstack((X, y[:, np.newaxis]))
np.save("Ecoli.npy", Data)

#Glass
Data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', dtype = float, delimiter = ",")
np.save("Glass.npy", Data)

#Letter Recognition
Data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data', delimiter = ",",  dtype = str)
X = Data[:, range(1, Data.shape[1])].astype(float)
y = np.unique(Data[:, 0], return_inverse=True)[1]
Data = np.hstack((X, y[:, np.newaxis]))
np.save("Letter-Recognition.npy", Data)
