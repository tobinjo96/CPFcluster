CPFcluster
========

An implementation of the Component-wise Peak-Finding (CPF) clustering method, presented in 'A Theoretical Analysis of Density Peaks Clustering and the CPF Algorithm'.

<p align="center" width="100%">
    <img width="100%" src="CPF.png">
</p>

<p align="center" width="100%">
    <img width="100%" src="CPF_grid.pdf">
</p>

Dependencies
------------

*CPFcluster* supports Python 3, with numpy, scipy, itertools, multiprocessing and scikit-learn. These should be linked with a BLAS implementation
(e.g., OpenBLAS, ATLAS, Intel MKL). 

Installation
------------

[CPFcluster](https://pypi.python.org/pypi/CPFcluster) is available on PyPI, the Python Package Index.

```sh
$ pip install CPFcluster
```

How To Use
----------

To use CPFcluster, first import the *CPFcluster* module.
```python
    from CPFcluster import CPFcluster
```    
### Clustering a Dataset

A CPFcluster object is constructed using the *fit* method, which returns a clustering of a dataset.
```python
    CPF = CPFcluster(k, rho, alpha, n_jobs, remove_duplicates, cutoff)
    CPF.fit(X)
```    
CPFcluster takes 6 arguments:

* **k** Number of nearest-neighbors used to create connected components from the dataset and compute the density.
* **rho** (Defaults to 0.4) Parameter used in threshold for center selection.
* **alpha** (Defaults to 1) Optional parameter used in threshold of edge weights for center selection, not discussed in paper.
* **n_jobs** (Defaults to 1) Number of cores for program to execute on. 
* **remove_duplicates** (Defaults to False) Option to remove duplicate rows from data in advance of clustering. 
* **cutoff** (Defaults to 1) Threshold for removing instances as outliers. Instances with fewer edges than the cutoff value are removed. 

The CPFcluster object is then fit to a dataset:
* **X** An *n-by-d* numpy.ndarray with training data. The rows correspond to *n* observations, and the columns
  correspond to *d* dimensions.

The result object further contains:
* **CCmat** An *n-by-n* sparse matrix representation of the *k*-NN graph.  
* **components** A vector containing the index of the component to which each instance belongs. If the instance is an outlying point, the value will be NaN. 
* **labels_** The final cluster labelings. 

Experimentation
-------
To replicate the experiments in the original paper:
```sh
python3 run_CPF.py
```

CPFcluster
-------

*CPFcluster* has an [MIT License](https://en.wikipedia.org/wiki/MIT_License).

See [LICENSE](LICENSE).


