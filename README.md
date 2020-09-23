[![Build Status](https://github.com/dstein64/aghasher/workflows/build/badge.svg)](https://github.com/dstein64/aghasher/actions)

CPFcluster
========

An implementation of the Component-wise Peak-Finding (CPF) clustering method, presented in ....

Dependencies
------------

*CPFcluster* supports Python 3, with numpy, scipy, itertools, multiprocessing and aghasher. These should be linked with a BLAS implementation
(e.g., OpenBLAS, ATLAS, Intel MKL). The package [aghasher](https://pypi.python.org/pypi/aghasher) is used to implement the k nearest neighbour graph approximation introduced in Zhang et. al. (2013). 

Installation
------------

[CPFcluster](https://pypi.python.org/pypi/CPFcluster) is available on PyPI, the Python Package Index.

```sh
$ pip install CPFcluster
```

How To Use
----------

To use CPFcluster, first import the *CPFcluster* module.

    import CPFcluster
    
### Clustering a Dataset

A CPFclustering is constructed using the *train* method, which returns an CPFclustering of a dataset.

    result = CPFcluster.CPFclustering.train(X, k, K, beta, reps, num_hashbits, blocksz, n_core)

CPFclustering.train takes 8 arguments:

* **X** An *n-by-d* numpy.ndarray with training data. The rows correspond to *n* observations, and the columns
  correspond to *d* dimensions.
* **k** Number of nearest-neighbors used to create connected components from the dataset.
* **K** Number of nearest-neighbors used to compute the local density of each instance.
* **beta** (optional; defaults to 30) Number of clusters to be tested for each component in the center selection method. 
* **reps** (optional; defaults to 50) Number of repetitions of the locality sensitive hashing method used in computing the k nearest-neighbor graphs. 
* **num_hashbits** (optional; defaults to 12) Number of hashbits used in locality sensitive hashing method. 
* **blocksz** (optional; defaults to 100) Size of the neighborhood on which brute force kNN is computed in locality sensitive hashing method. 
* **n_core** (optional; defaults to 1) Number of processors to be used when computing nearest-neighbor graph. If set to 1, parallel processing does not take place. 


Tests
-----


CPFcluster
-------

*CPFcluster* has an [MIT License](https://en.wikipedia.org/wiki/MIT_License).

See [LICENSE](LICENSE).

References
----------
Zhang, Yan-Ming, et al. “Fast kNN graph construction with locality sensitive hashing.“ Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2013.

Liu, Wei, Jun Wang, Sanjiv Kumar, and Shih-Fu Chang. 2011. “Hashing with Graphs.” In Proceedings of the 28th
International Conference on Machine Learning (ICML-11), edited by Lise Getoor and Tobias Scheffer, 1–8. ICML ’11. New
York, NY, USA: ACM.
