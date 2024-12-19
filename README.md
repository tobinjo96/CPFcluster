# CPFcluster: the Component-wise Peak-Finding algorithm

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
    min_samples=5,                   # minimum number of neighbors to consider for connectivity #
    rho=[0.3, 0.5, 0.7],             # parameter that controls the number of clusters for each component set #
    alpha=[0.8, 1.0, 1.2],           # parameter for edge-cutoff in cluster detection #
    n_jobs=1,                        # number of parallel jobs for computation #
    cutoff=1,                        # threshold for filtering out small connected components as outliers #
    merge=False,                     # whether to merge similar clusters based on thresholds #
    merge_threshold=0.5,             # distance threshold for merging clusters #
    density_ratio_threshold=0.1,     # density ratio threshold for merging clusters #
    distance_metric='euclidean',     # metric for distance computation (e.g., 'euclidean', 'manhattan', 'cosine') #
    remove_duplicates=False,         # whether to remove duplicate data points before clustering #
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
  *Default*: `[0.3, 0.5, 0.7]`, a list of `rho` values for grid search  

- **`alpha`** *(list)*:  
  An optional parameter used to set the threshold for edge weights during center selection, not discussed in the paper.  
  *Default*: `[0.8, 1.0, 1.2]`, a list of `alpha` values for grid search

- **`n_jobs`** *(int)*:  
  Number of parallel jobs for computation. Specify `n_jobs=-1` (and include the `__name__ == "__main__":` line in your script) to use all cores.  
  *Default*: `1`
  
- **`cutoff`** *(int)*:  
  In the mutual k-NN graph, vertices with a number of edges less than or equal to the specified `cutoff` value are identified as outliers.  
  *Default*: `1` 
  
- **`merge`** *(bool)*:  
  Specifies whether to merge clusters that are similar based on distance and density-ratio thresholds. Two clusters will be merged only if the distance between their centroids is less than the `merge_threshold` AND the density ratio exceeds the `density_ratio_threshold`.    
  *Default*: `False` 
  
- **`merge_threshold`** *(float)*:  
  The distance threshold that determines whether two clusters should be merged. Clusters will be merged if the distance between their centroids is less than the `merge_threshold`. This parameter helps to combine clusters that are close in the feature space, potentially reducing over-segmentation. A range of 0.1–1.0 works well across diverse datasets (after standardization).  
  *Default*: `0.5`  

- **`density_ratio_threshold`** *(float)*:  
  The density ratio threshold that determines whether two clusters should be merged. Clusters are merged if the ratio of densities between two clusters (lower density/higher density) exceeds the `density_ratio_threshold`, ensuring that only clusters with comparable densities are merged. A range of 0.1–0.5 is observed to work well across various datasets (after standardization).   
  *Default*: `0.1`  
  
- **`distance_metric`** *(str)*:  
  Metric to use for distance computation. Options include:  
  - `'euclidean'`: Euclidean distance (default).  
  - `'manhattan'`: sum of absolute differences.  
  - `'cosine'`: cosine similarity-based distance.  
  - `'chebyshev'`: maximum difference along any dimension.  
  - `'minkowski'`: generalized distance metric requiring a parameter \(p\) (e.g., \(p=1\) for Manhattan, \(p=2\) for Euclidean).  
  - `'hamming'`: fraction of differing attributes between samples (useful for binary data).  
  - `'jaccard'`: used for binary attributes to measure similarity based on set intersection and union.  

- **`remove_duplicates`** *(bool)*:  
  Whether to remove duplicate data points before clustering.  
  *Default*: `False`

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

## Code Example 1: Clustering with the Dermatology Dataset
The script below demonstrates how to use CPFcluster with the Dermatology dataset, available in the Data folder.  

```python

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from core import CPFcluster


# Define the main function to utilize Python's multiprocessing unit (in Windows OS).
def main():
    # Step 1: Load the Data
    
    # Load the Dermatology dataset
    Data = np.load("Data/dermatology.npy")
    X = Data[:,:-1]  # Feature data
    y = Data[:,-1]  # True labels (used here for evaluation, not clustering)

    # Normalize dataset for easier hyperparameter tuning
    X = StandardScaler().fit_transform(X)
    
    
    # Step 2: Initialize CPFcluster
    cpf = CPFcluster(
        min_samples=10,
        rho=0.5,
        alpha=1.0,
        merge=True,
        merge_threshold=0.6,
        n_jobs=-1,
        plot_tsne=True,
        plot_pca=True,
        plot_umap=True
    )
    
    
    # Step 3: Fit the Model
    cpf.fit(X)

    # access the cluster labels
    print("Cluster labels:", cpf.labels)
    
    
    # Step 4: Calculate Cluster Validity Indices
    ari = adjusted_rand_score(y, cpf.labels)
    print(f"Adjusted Rand Index (ARI): {ari:.2f}")


if __name__ == "__main__":
    main()


```


### Visualize Results

If `plot_tsne`, `plot_pca`, or `plot_umap` was set to `True`, clustering results will be visualized automatically after fitting the model. Here are the PCA, UMAP, and t-SNE visualizations for the Dermatology dataset:

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



## Code Example 2: Grid Search for Optimal Parameter Configuration.
The script below demonstrates how to tune the parameters to obtain the highest Calinski-Harabasz score.  

```python

import os
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
from itertools import product
from time import time
from core import CPFcluster  
from tqdm import tqdm

def main():
    # Define datasets and parameter ranges
    datasets = ["seeds", "glass", "vertebral", "ecoli", "dermatology"]

    # Parameter ranges for grid search
    min_samples_range = [3, 5, 10]
    rho_range = [0.1, 0.4, 0.7]
    alpha_range = [0.5, 1, 1.5]
    cutoff_range = [1, 2, 3]
    merge_threshold_range = [0.3, 0.5, 0.7]
    density_ratio_threshold_range = [0.05, 0.1, 0.2]

    # Directory for saving results
    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_path = os.path.join(results_dir, "CPFcluster_grid_search_results.csv")
    
    # Write header to results CSV
    with open(results_path, 'w', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(["Dataset", "min_samples", "rho", "alpha", "cutoff", 
                         "merge_threshold", "density_ratio_threshold", 
                         "CH_Index", "Time"])

    # Iterate through datasets
    for dataset in datasets:
        # Load dataset
        Data = np.load(f"Data/{dataset}.npy")  
        X = Data[:, :-1]
        y = Data[:, -1]
        
        # Normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # Parameter grid
        param_grid = product(min_samples_range, rho_range, alpha_range, 
                             cutoff_range, merge_threshold_range, 
                             density_ratio_threshold_range)

        for params in tqdm(param_grid, desc=f"Processing {dataset}"):
            min_samples, rho, alpha, cutoff, merge_threshold, density_ratio_threshold = params
            start_time = time()

            # Initialize CPFcluster with current parameters
            model = CPFcluster(
                min_samples=min_samples,
                rho=rho,
                alpha=alpha,
                cutoff=cutoff,
                merge=True,
                merge_threshold=merge_threshold,
                density_ratio_threshold=density_ratio_threshold,
                n_jobs=-1
            )
            
            # Fit model
            model.fit(X)
            labels = model.labels

            # Skip iteration if all points are assigned to one cluster
            if len(np.unique(labels)) < 2:
                continue

            # Compute the adjusted Calinski-Harabasz index
            ch_index = calinski_harabasz_score(X, labels)
            elapsed_time = time() - start_time

            # Write results to CSV
            with open(results_path, 'a', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerow([dataset, min_samples, rho, alpha, cutoff, 
                                 merge_threshold, density_ratio_threshold, 
                                 ch_index, elapsed_time])

if __name__ == "__main__":
    main()

```


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
   Offers flexibility through the `distance_metric` parameter, which supports multiple distance measures (e.g., `'euclidean'`, `'manhattan'`, `'cosine'`).

6. **Visualization Support**:  
   Includes built-in options for t-SNE, PCA, and UMAP visualizations to enhance interpretability of clustering results.


## Limitations

1. **Parameter Sensitivity**:  
   The results can be sensitive to the choice of `rho`, `alpha`, and `cutoff`, which require careful tuning for optimal results.

2. **Computational Overhead**:  
   Computing the nearest-neighbor graph for very large datasets can demand significant memory and processing resources.

