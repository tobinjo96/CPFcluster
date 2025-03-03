# CPFcluster: the Component-wise Peak-Finding algorithm

### Note that this version differs from the PyPI release (Dec 6, 2024): certain arguments now accept lists instead of scalars, eliminating the need for users to write loops.



Illustration of the **CPFcluster** algorithm. **(1)** The mutual k-NN graph is constructed, from which we extract two component sets. **(2)** Densities are computed as the inverse of the distance from an instance to its k-th nearest neighbor (darker color represents higher density). The distance from each instance to its nearest neighbor of higher density is found (larger point represents larger distance to point of higher density). The peak-finding criterion is the product of these two quantities. **(3)** For each connected component, density-level sets are used to assess potential cluster centers, shown in yellow. **(4)** Non-center instances are assigned to the same cluster as their nearest neighbor of higher local density. A sample assignment path is shown in gold for the purple cluster.


<img src="Plots\CPF_Illustration.png">


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


## Class: CPFcluster

**CPFcluster** is a scalable and flexible density-based clustering method that integrates the strengths of density-level set and mode-seeking approaches. This combination offers several advantages, including: (1) the ability to detect outliers, (2) effective identification of clusters with varying densities and overlapping regions, and (3) robustness against spurious density maxima. The `CPFcluster` class is designed to identify outliers, merge similar clusters, and visualize clustering results in 2D using techniques like PCA, UMAP and t-SNE.
```python
CPFcluster(
    min_samples=5,                   # the k (number of neighbors) in the k-NN graph #
    rho=None,                        # parameter that controls the number of clusters for each component set #
    alpha=None,                      # parameter for edge-cutoff in cluster detection #
    n_jobs=1,                        # number of parallel jobs for computation #
    remove_duplicates=False,         # whether to remove duplicate data points before clustering #
    cutoff=1,                        # threshold for filtering out small connected components as outliers #
    distance_metric='euclidean',     # metric for distance computation (e.g., 'euclidean', 'manhattan', 'cosine') #
    merge=False,                     # whether to merge similar clusters based on thresholds #
    merge_threshold=None,            # distance threshold for merging clusters #
    density_ratio_threshold=None,    # density ratio threshold for merging clusters #
    plot_umap=False,                 # whether to plot UMAP visualization after clustering #
    plot_pca=False,                  # whether to plot PCA visualization after clustering #
    plot_tsne=False                  # whether to plot t-SNE visualization after clustering #
)
```

### Parameters

- **`min_samples`** *(int)*:  
Number of nearest-neighbors used to create connected components from the dataset and compute the density. This parameter is used in the `build_CCgraph` function to construct the k-NN graph and extract the component sets. The default value is `5`, but a value `10` normally works well.  
 *Default*: `5`

- **`rho`** *(list)*:  
  The `rho` parameter in Definition 10 of the paper "A Theoretical Analysis of Density Peaks Clustering and the Component-Wise Peak-Finding Algorithm". Varying the parameter `rho` determines the number of clusters for each component set.  
  *Default*: `[0.4]`, if users not specify any value    

- **`alpha`** *(list)*:  
  An optional parameter used to set the threshold for edge weights during center selection, not discussed in the paper.  
  *Default*: `[1]`, if users not specify any value

- **`n_jobs`** *(int)*:  
  Number of parallel jobs for computation. Specify `n_jobs=-1` (and include the `__name__ == "__main__":` line in your script) to use all cores.  
  *Default*: `1`
  
- **`remove_duplicates`** *(bool)*:  
  Whether to remove duplicate data points before clustering.  
  *Default*: `False`

- **`cutoff`** *(int)*:  
  In the mutual k-NN graph, vertices with a number of edges less than or equal to the specified `cutoff` value are identified as outliers.  
  *Default*: `1` 
  
- **`distance_metric`** *(str)*:  
  Metric to use for distance computation. Options include:  
  - `'euclidean'`: Euclidean distance (default).  
  - `'manhattan'`: sum of absolute differences.  
  - `'cosine'`: cosine similarity-based distance.  
  - `'chebyshev'`: maximum difference along any dimension.  
  - `'minkowski'`: generalized distance metric requiring a parameter \(p\) (e.g., \(p=1\) for Manhattan, \(p=2\) for Euclidean).  
  - `'hamming'`: fraction of differing attributes between samples (useful for binary data).  
  - `'jaccard'`: used for binary attributes to measure similarity based on set intersection and union.  

- **`merge`** *(bool)*:  
  Specifies whether to merge clusters that are similar based on distance and density-ratio thresholds. Two clusters will be merged only if the distance between their centroids is less than the `merge_threshold` AND the density ratio exceeds the `density_ratio_threshold`.    
  *Default*: `False` 
  
- **`merge_threshold`** *(list)*:  
  The distance threshold that determines whether two clusters should be merged. Clusters will be merged if the distance between their centroids is less than the `merge_threshold`. This parameter helps to combine clusters that are close in the feature space, potentially reducing over-segmentation. A range of 0.1–1.0 works well across diverse datasets (after standardization).  
  *Default*: `[0.5]`, if users not specify any value    

- **`density_ratio_threshold`** *(list)*:  
  The density ratio threshold that determines whether two clusters should be merged. Clusters are merged if the ratio of densities between two clusters (lower density/higher density) exceeds the `density_ratio_threshold`, ensuring that only clusters with comparable densities are merged. A range of 0.1–0.5 is observed to work well across various datasets (after standardization).   
  *Default*: `[0.1]`, if users not specify any value    
  
- **`plot_clusters_umap`** *(bool)*:  
  Whether to visualize the clusters via UMAP.  
  *Default*: `False`

- **`plot_clusters_pca`** *(bool)*:  
  Whether to visualize the clusters via PCA.  
  *Default*: `False`
  
- **`plot_clusters_tsne`** *(bool)*:  
  Whether to visualize the clusters via t-SNE.  
  *Default*: `False`


### Methods

- `fit(X)`: Apply the CPF method to the input data X. <br>  
  - `X` *(np.ndarray)*: input data of shape `(n_samples, n_features)`.
  - **Returns**:
    - None. Update the instance attributes with identified cluster labels. Outliers are labeled as `-1`.

- `calculate_centroids_and_densities(X, labels)`: Calculates the centroid and average density of each cluster.<br>  
  - `X` *(np.ndarray)*: input data of shape `(n_samples, n_features)`.
  - `labels` *(np.ndarray)*: cluster labels of the input data.
  - **Returns**:
    - `centroids` *(np.ndarray)*: centroids of the clusters.
    - `densities` *(np.ndarray)*:  average density of each cluster.

- `merge_clusters(X, centroids, densities, labels)`: Merges similar clusters based on the distance and density-ratio thresholds.<br>  
  - `X` *(np.ndarray)*: input data of shape `(n_samples, n_features)`.
  - `centroids` *(np.ndarray)*: centroids of the clusters.
  - `densities` *(np.ndarray)*:  average density of each cluster.
  - `labels` *(np.ndarray)*: cluster label for each sample. 
  - **Returns**: 
    - `labels` *(np.ndarray)*: updated cluster labels after merging.
    

- `cross_validate(self, X, validation_index=calinski_harabasz_score)`: Find the best parameter configuration according to the user-specified clustering evaluation metric.<br>  
  - `X` *(np.ndarray)*: input data of shape `(n_samples, n_features)`.
  - `validation_index` *(callable)*: a clustering metric from sklearn.metrics
  - **Returns**: 
    - `best_params` *(tuple)*: best parameter configuration (`min_samples`, `rho`, `alpha`, `merge_threshold`, `density_ratio_threshold`).
    - `best_score` *(float)*: the value of the clustering metric evaluated at the best parameter configuration.


- `plot_results(self, X, k=None, rho=None, alpha=None, merge_threshold=None, density_ratio_threshold=None)`: Produce the PCA, UMAP and t-SNE plots.<br>  
  - `X` *(np.ndarray)*: input data of shape `(n_samples, n_features)`.
  - ... 
  - **Returns**: 
    - `None`.



---

## Helper Functions

### `build_CCgraph(X, min_samples, cutoff, n_jobs, distance_metric='euclidean')`
Construct the k-NN graph and extract the connected components.

- **Parameters**:
  - **`X`** *(np.ndarray)*: input data of shape `(n_samples, n_features)`.
  - ...

- **Returns**:
  - **`components`** *(np.ndarray)*: connected component for each sample. If a sample is an outlier, the value will be NaN.
  - **`CCmat`** *(scipy.sparse.csr_matrix)*: an n-by-n sparse matrix representation of the k-NN graph.
  - **`knn_radius`** *(np.ndarray)*: distance to the min_samples-th (i.e., k-th) neighbor for each sample.


### `get_density_dists_bb(X, k, components, knn_radius, n_jobs)`
Identify the "big brother" (nearest neighbor of higher density) for each sample.

- **Parameters**:
  - **`X`** *(np.ndarray)*: input data of shape `(n_samples, n_features)`.
  - **`k`** *(int)*: identical to `min_samples`, the neighborhood parameter \(k\) in the mutual k-NN graph.
  - ...

- **Returns**:
  - **`best_distance`** *(np.ndarray)*: best distance for each sample.
  - **`big_brother`** *(np.ndarray)*: index of the "big brother" for each sample.


### `get_y(CCmat, components, knn_radius, best_distance, big_brother, rho, alpha, d)`
Assigns cluster labels to data points based on density and connectivity properties.

- **Parameters**:
  - ...
  - **`d`** *(int)*: dimension of the input data `X`.

- **Returns**:
  - **`y_pred`** *(np.ndarray)*: identified cluster labels for the input data.

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

### Time Complexity
In the provided code example, the CPFcluster algorithm is executed 768 times for each dataset (calculated as 3×4×4×4×4=768). While CPFcluster is generally fast, performing 768 repetitions can take a few minutes for small datasets and several hours for large datasets. For reference, on a computer with an Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz, the execution time is 453 seconds for the Ecoli dataset and 944 seconds for the Dermatology dataset.

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




---

## Data

The data folder contains the following datasets for testing and benchmarking different clustering methods. The following datasets are from the UCI Machine Learning Repository:

- **Dermatology**: The Dermatology dataset for classifying erythemato-squamous diseases.

- **Ecoli**: This dataset predicts the localization sites of proteins within E. coli cells.

- **Glass Identification**: This dataset is used for classifying glass types for forensic analysis.

- **HTRU2**: HTRU2 distinguishes pulsar candidates from non-pulsar stars based on astronomical data.

- **Letter Recognition**: The dataset is designed to classify uppercase English alphabet letters.

- **MAGIC Gamma Telescope**: This dataset predicts whether detected particles are high-energy gamma rays.

- **Optical Recognition of Handwritten Digits**: This dataset is used for classifying handwritten digit images.

- **Page Blocks Classification**: Classify blocks of text in scanned documents into various categories.

- **Pen-Based Recognition of Handwritten Digits**: The dataset is for digit classification based on pen movement data.

- **Seeds**: This dataset classifies wheat seed varieties based on geometric properties.

- **Vertebral Column**: The dataset is used to classify conditions of vertebral disks as normal or abnormal.

The following two datasets are from Kaggle: **Fraud Detection Bank** and **Paris Housing Classification**. The **Phoneme** dataset is from R (https://r-packages.io/datasets/phoneme).  


---

## Key Features


1. **Outlier Detection**:  
   The algorithm identifies and excludes small connected components as outliers based on the `cutoff` parameter.

2. **Cluster Merging**:  
   Merges similar clusters by evaluating proximity (`merge_threshold`) and density similarity (`density_ratio_threshold`).

4. **Scalability**:  
   Supports parallel processing with the `n_jobs` parameter, enabling efficient computation for large datasets.

5. **Customizable Metrics**:  
   Offers flexibility through the `distance_metric` parameter, which supports multiple distance measures (e.g., `'euclidean'`, `'manhattan'`, `'cosine'`), and the `validation_index` parameter, which supports multiple internal clustering validation metrics (e.g., `'silhouette_score'`, `'calinski_harabasz_score'`, `'davies_bouldin_score'`).

6. **Visualization Support**:  
   Includes built-in options for t-SNE, PCA, and UMAP visualizations to enhance interpretability of clustering results.

