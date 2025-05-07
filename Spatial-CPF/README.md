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

## (1) The `create_neighbor_adjacency_matrix` function accepts a two-column numpy array, where the first column is latitude and the second column is longitude.

- `create_neighbor_adjacency_matrix(coords, n_neighbors)`: Create a 0/1 adjacency matrix where 1 indicates that a point is among the k-nearest neighbors of another point.<br>  
  - `coords` *(np.ndarray)*: numpy array of shape `(n_samples, 2)` with latitude/longitude in degrees.
  - `n_neighbors` *(int)*: number of nearest neighbors to consider. In general, `n_neighbors` will have the same value as the argument `min_samples` in the `build_CCgraph` function.
  - **Returns**:
    - `adjacency_matrix`: binary numpy array of shape `(n_samples, n_samples)`.


## (2) The `fit` function has one additional argument: `geo_neighbor_adjacency_matrix`.

- `fit(X, geo_neighbor_adjacency_matrix)`: Apply the CPF method to the input data X, incorporating geographic proximity via a precomputed adjacency matrix. <br>  
  - `X` *(np.ndarray)*: input data of shape `(n_samples, n_features)`.
  - `geo_neighbor_adjacency_matrix` *(np.ndarray)*: binary numpy array of shape `(n_samples, n_samples)`.
  - **Returns**:
    - None. Update the instance attributes with identified cluster labels. Outliers are labeled as `-1`.


## (3) The `build_CCgraph` function has one additional argument: `geo_neighbor_adjacency_matrix`. 

- `build_CCgraph(X, geo_neighbor_adjacency_matrix, min_samples, cutoff, n_jobs, distance_metric='euclidean')`: Constructs a second adjacency matrix based on feature similarity in X, and combines it with the geographic adjacency matrix using element-wise multiplication. Two samples are considered neighbors only if they are both similar in features and geographically close. The resulting matrix is used to extract connected components. <br>  


---

## Example Usage

To apply Spatial-CPF:
- Use `create_neighbor_adjacency_matrix` to generate the spatial adjacency matrix.
- Pass the matrix to the fit method to perform clustering.

A complete example using a geochemical dataset is provided in Code_Example_Geo.py. For detailed methodology and analysis results, refer to our [technical report](https://arxiv.org/abs/2505.00510)


