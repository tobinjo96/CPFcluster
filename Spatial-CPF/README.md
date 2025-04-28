# The code in this folder is for the cluster analysis of data that include geographic coordinates. 


If you used this package in your research, please cite it:
```latex
@ARTICLE{10296014,
  author={Tobin, Joshua and Zhang, Mimi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A Theoretical Analysis of Density Peaks Clustering and the Component-Wise Peak-Finding Algorithm}, 
  year={2024},
  volume={46},
  number={2},
  pages={1109-1120},
  doi={10.1109/TPAMI.2023.3327471}}
```



---

## The `create_neighbor_adjacency_matrix` function accepts a two-column numpy array, where the first column is latitude and the second column is longitude.

- `create_neighbor_adjacency_matrix(coords, n_neighbors=15)`: Create a 0/1 adjacency matrix where 1 indicates that a point is among the k-nearest neighbors of another point.<br>  
  - `coords` *(np.ndarray)*: numpy array of shape `(n_samples, 2)` with latitude/longitude in degrees.
  - `n_neighbors` *(int)*: number of nearest neighbors to consider.
  - **Returns**:
    - `adjacency_matrix`: binary numpy array of shape `(n_samples, n_samples)`.


## The `fit` function has one additional argument: `geo_neighbor_adjacency_matrix`.

- `fit(X, geo_neighbor_adjacency_matrix)`: Apply the CPF method to the input data X. <br>  
  - `X` *(np.ndarray)*: input data of shape `(n_samples, n_features)`.
  - `geo_neighbor_adjacency_matrix` *(np.ndarray)*: binary numpy array of shape `(n_samples, n_samples)`.
  - **Returns**:
    - None. Update the instance attributes with identified cluster labels. Outliers are labeled as `-1`.


## The `build_CCgraph` function has one additional argument: `geo_neighbor_adjacency_matrix`.

- `build_CCgraph(X, geo_neighbor_adjacency_matrix, min_samples, cutoff, n_jobs, distance_metric='euclidean')`: Construct the k-NN graph and extract the connected components. <br>  



---
## The `Code Example Geo.py` file provides ... We gave a detailed explanation on the data ... here ...

