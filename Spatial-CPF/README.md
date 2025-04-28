# CPFcluster: the Component-wise Peak-Finding algorithm

## This branch is for the cluster analysis of data that include geographic coordinates. 


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
## Functions differ from the ones in the main branch

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
## Code Example: Clustering with the Dermatology Dataset
The script below demonstrates how to use CPFcluster with the Dermatology dataset, available in the Data folder.  

```python

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from core import CPFcluster


# Define the main function to utilize Python's multiprocessing unit (in Windows OS).
def main():
    # Load the dataset.
    Data = np.load("Data/dermatology.npy")
    X = Data[:, :-1]
    y = Data[:, -1]   # true labels (used here for evaluation, not clustering)

    # Normalize dataset for easier hyperparameter tuning.
    X = StandardScaler().fit_transform(X)

    # Initialize CPFcluster with multiple rho, alpha, merge_threshold and density_ratio_threshold values.
    cpf = CPFcluster(
        min_samples=10,
        rho=[0.3, 0.5, 0.7, 0.9],  # list of rho values for grid search
        alpha=[0.6, 0.8, 1.0, 1.2],  # list of alpha values for grid search
        merge=True,
        merge_threshold=[0.6, 0.5, 0.4, 0.3],  # list of merge thresholds
        density_ratio_threshold=[0.1, 0.2, 0.3, 0.4],  # list of density ratio thresholds
        n_jobs=-1,
        plot_tsne=True,
        plot_pca=True,
        plot_umap=True
    )

    # Fit the model for a range of min_samples values.
    print("Fitting CPFcluster...")
    cpf.fit(X, k_values=[5, 10, 15])

    # Perform cross-validation to find the best (min_samples, rho, alpha, merge_threshold, density_ratio_threshold)
    print("Performing cross-validation...")
    best_params, best_score = cpf.cross_validate(X, validation_index=calinski_harabasz_score)
    print(f"Best Parameters: min_samples={best_params[0]}, rho={best_params[1]:.2f}, alpha={best_params[2]:.2f}, "
        f"merge_threshold={best_params[3]:.2f}, density_ratio_threshold={best_params[4]:.2f}. "
        f"Best Validation Score (Calinski-Harabasz Index): {best_score:.2f}")


    # Access the cluster labels for the best paramter configuration.
    best_labels = cpf.clusterings[best_params]
    print("Cluster labels for best parameters:", best_labels)

    # Evaluate the clustering performance using Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(y, best_labels)
    print(f"Adjusted Rand Index (ARI) for best parameters: {ari:.2f}")

    # Plot results for the best paramter configuration.
    print("Plotting results...")
    cpf.plot_results(
        X,
        k=best_params[0],
        rho=best_params[1],
        alpha=best_params[2],
        merge_threshold=best_params[3],
        density_ratio_threshold=best_params[4]
    )


if __name__ == "__main__":
    main()


```

### Visualize Results

If `plot_tsne`, `plot_pca`, or `plot_umap` was set to `True`, clustering results will be visualized automatically after fitting the model. Here are the PCA, UMAP, and t-SNE visualizations for the Dermatology dataset (`min_samples=15, rho=0.30, alpha=0.60, merge_threshold=0.60, density_ratio_threshold=0.10`):

<table width="100%">
  <tr>
    <td align="center"><strong>PCA Plot</strong></td>
    <td align="center"><strong>UMAP Plot</strong></td>
    <td align="center"><strong>t-SNE Plot</strong></td>
  </tr>
  <tr>
    <td>
      <img src="Plots\PCA Plot Derma.png" alt="PCA Plot" width="100%">
    </td>
    <td>
      <img src="Plots\UMAP Plot Derma.png" alt="UMAP Plot" width="100%">
    </td>
    <td>
      <img src="Plots\TSNE Plot Derma.png" alt="t-SNE Plot" width="100%">
    </td>
  </tr>
</table>

