import numpy as np
import scipy.sparse
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors
import utils
import gc
import itertools
from plotting import plot_clusters_tsne, plot_clusters_pca,plot_clusters_umap
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in cast")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

def build_CCgraph(X, min_samples, cutoff, n_jobs, distance_metric='euclidean'):
    """
    Constructs a connected component graph (CCgraph) for input data using k-nearest neighbors.
    Identifies connected components and removes outliers based on a specified cutoff.
    
    Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        min_samples (int): Minimum number of neighbors to consider for connectivity.
        cutoff (int): Threshold for filtering out small connected components as outliers.
        n_jobs (int): Number of parallel jobs for computation.
        distance_metric (str): Metric to use for distance computation.
        
    Returns:
        components (np.ndarray): Array indicating the component each sample belongs to.
        CCmat (scipy.sparse.csr_matrix): Sparse adjacency matrix representing connections.
        knn_radius (np.ndarray): Array of distances to the min_samples-th neighbor for each sample.
    """
    n = X.shape[0]
    kdt = NearestNeighbors(n_neighbors=min_samples, metric=distance_metric, n_jobs=n_jobs, algorithm='auto').fit(X)
    CCmat = kdt.kneighbors_graph(X, mode='distance').astype(np.float32)
    distances, _ = kdt.kneighbors(X)
    knn_radius = distances[:, min_samples - 1]
    CCmat = CCmat.minimum(CCmat.T)

    # Remove outlying points
    _, components = scipy.sparse.csgraph.connected_components(CCmat, directed=False, return_labels=True)
    comp_labs, comp_count = np.unique(components, return_counts=True)
    outlier_components = comp_labs[comp_count <= cutoff]
    nanidx = np.isin(components, outlier_components)

    # Change components to float to accommodate NaN values
    components = components.astype(np.float32)  
    components[nanidx] = -1 

    return components, CCmat, knn_radius


def get_density_dists_bb(X, k, components, knn_radius, n_jobs):
    """
    Computes the best distance for each data point and identifies its 'big brother',
    which is the nearest point with a greater density.
    
    Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        k (int): Number of neighbors to consider for density estimation.
        components (np.ndarray): Array indicating the component each sample belongs to.
        knn_radius (np.ndarray): Array of distances to the min_samples-th neighbor for each sample.
        n_jobs (int): Number of parallel jobs for computation.
        
    Returns:
        best_distance (np.ndarray): Array of best distances for each data point.
        big_brother (np.ndarray): Array indicating the index of the 'big brother' for each point.
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif n_jobs <= 0:
        n_jobs = 1
    best_distance = np.full((X.shape[0]), np.nan, dtype=np.float32)
    big_brother = np.full((X.shape[0]), np.nan, dtype=np.int32)
    comps = np.unique((components[~np.isnan(components)])).astype(np.int32)
    ps = np.zeros((1, 2), dtype=np.float32)
    
    for cc in comps:
        cc_idx = np.where(components == cc)[0].astype(np.int32)
        nc = len(cc_idx)
        kcc = max(1,min(k, nc - 1))
        kdt = NearestNeighbors(n_neighbors=kcc, metric='euclidean', n_jobs=n_jobs, algorithm='kd_tree').fit(X[cc_idx, :])
        distances, neighbors = kdt.kneighbors(X[cc_idx, :])
        cc_knn_radius = knn_radius[cc_idx]
        cc_best_distance = np.empty((nc), dtype=np.float32)
        cc_big_brother = np.empty((nc), dtype=np.int32)
        cc_radius_diff = cc_knn_radius[:, np.newaxis] - cc_knn_radius[neighbors]
        
        rows, cols = np.where(cc_radius_diff > 0)
        rows, unidx = np.unique(rows, return_index=True)
        del cc_radius_diff
        gc.collect()
        
        cols = cols[unidx]
        cc_big_brother[rows] = neighbors[rows, cols]
        cc_best_distance[rows] = distances[rows, cols]
        
        search_idx = np.setdiff1d(np.arange(X[cc_idx, :].shape[0], dtype=np.int32), rows)
        ps = np.vstack((ps, [len(cc_idx), len(search_idx) / len(cc_idx)]))
        
        for indx_chunk in utils.chunks(search_idx, 100):
            search_radius = cc_knn_radius[indx_chunk]
            GT_radius = cc_knn_radius < search_radius[:, np.newaxis]
            
            if np.any(np.sum(GT_radius, axis=1) == 0):
                max_i = [i for i in range(GT_radius.shape[0]) if np.sum(GT_radius[i, :]) == 0]
                if len(max_i) > 1:
                    for max_j in max_i[1:]:
                        GT_radius[max_j, indx_chunk[max_i[0]]] = True
                max_i = max_i[0]
                cc_big_brother[indx_chunk[max_i]] = indx_chunk[max_i]
                cc_best_distance[indx_chunk[max_i]] = np.inf
                indx_chunk = np.delete(indx_chunk, max_i)
                GT_radius = np.delete(GT_radius, max_i, 0)
            
            GT_distances = ([X[cc_idx[indx_chunk[i]], np.newaxis], X[cc_idx[GT_radius[i, :]], :]] for i in range(len(indx_chunk)))
            
            if GT_radius.shape[0] > 50:
                try:
                    pool = mp.Pool(processes=n_jobs)
                    N = 25
                    distances = []
                    while True:
                        distance_comp = pool.map(utils.density_broad_search_star, itertools.islice(GT_distances, N))
                        if distance_comp:
                            distances.append(distance_comp)
                        else:
                            break
                    distances = [dis_pair for dis_list in distances for dis_pair in dis_list]
                    argmin_distance = [np.argmin(l) for l in distances]
                    pool.terminate()
                except Exception as e:
                    print("POOL ERROR:", e)
                    pool.close()
                    pool.terminate()
            else:
                distances = list(map(utils.density_broad_search_star, list(GT_distances)))
                argmin_distance = [np.argmin(l) for l in distances]
            
            for i in range(GT_radius.shape[0]):
                cc_big_brother[indx_chunk[i]] = np.where(GT_radius[i, :] == 1)[0][argmin_distance[i]]
                cc_best_distance[indx_chunk[i]] = distances[i][argmin_distance[i]]
        
        big_brother[cc_idx] = cc_idx[cc_big_brother.astype(np.int32)]
        best_distance[cc_idx] = cc_best_distance
    
    return best_distance, big_brother

def get_y(CCmat, components, knn_radius, best_distance, big_brother, rho, alpha, d):
    """
    Assigns cluster labels to data points based on density and connectivity properties.
    Identifies peaks within each connected component to create clusters.
    
    Parameters:
        CCmat (scipy.sparse.csr_matrix): Sparse adjacency matrix representing connections.
        components (np.ndarray): Array indicating the component each sample belongs to.
        knn_radius (np.ndarray): Array of distances to the min_samples-th neighbor for each sample.
        best_distance (np.ndarray): Array of best distances for each data point.
        big_brother (np.ndarray): Array indicating the index of the 'big brother' for each point.
        rho (float): Density parameter controlling the radius cutoff.
        alpha (float): Parameter for edge-cutoff in cluster detection.
        d (int): Number of features (dimensions) in the data.
        
    Returns:
        y_pred (np.ndarray): Array of predicted cluster labels.
    """
    n = components.shape[0]
    y_pred = np.full(n, -1)
    valid_indices = components != -1
    peaks = []
    n_cent = 0
    comps = np.unique(components[~np.isnan(components)]).astype(int)
    
    for cc in np.unique(components[valid_indices]):
        cc_idx = np.where(components == cc)[0]
        nc = cc_idx.size
        cc_knn_radius = knn_radius[cc_idx]
        cc_best_distance = best_distance[cc_idx]
        
        # Convert big_brother to cc_big_brother
        index = np.argsort(cc_idx)
        cc_big_brother = np.take(index, np.searchsorted(cc_idx[index], big_brother[cc_idx], sorter=index), mode='clip')
        
        not_tested = np.ones(nc, dtype=bool)
        peaked = np.divide(cc_best_distance, cc_knn_radius, where=cc_knn_radius != 0)
        peaked[(cc_best_distance == 0) & (cc_knn_radius == 0)] = np.inf
        
        cc_centers = [np.argmax(peaked)]
        not_tested[cc_centers[0]] = False
        
        CCmat_level = CCmat[cc_idx, :][:, cc_idx] 
        
        while np.sum(not_tested) > 0:  
            prop_cent = np.argmax(peaked[not_tested])
            prop_cent = np.arange(peaked.shape[0])[not_tested][prop_cent]
            
            if cc_knn_radius[prop_cent] > max(cc_knn_radius[~not_tested]):
                cc_level_set = np.where(cc_knn_radius <= cc_knn_radius[prop_cent])[0]
                CCmat_check = CCmat_level[cc_level_set, :][:, cc_level_set]
                n_cc, _ = scipy.sparse.csgraph.connected_components(CCmat_check, directed=False, return_labels=True)
                if n_cc == 1:
                    break
            
            # Calculate v_cutoff and e_cutoff
            v_cutoff = cc_knn_radius[prop_cent] / (rho**(1/d))
            e_cutoff = cc_knn_radius[prop_cent] / alpha
            
            e_mask = np.abs(CCmat_level.data) > e_cutoff
            CCmat_level.data[e_mask] = 0
            CCmat_level.eliminate_zeros()
            
            cc_cut_idx = np.where(cc_knn_radius < v_cutoff)[0] if cc_knn_radius[prop_cent] > 0 else np.where(cc_knn_radius <= v_cutoff)[0]
            reduced_CCmat = CCmat_level[cc_cut_idx, :][:, cc_cut_idx]
            
            _, cc_labels = scipy.sparse.csgraph.connected_components(reduced_CCmat, directed=False, return_labels=True)
            
            gc.collect()  # Manage memory

            center_comp = np.unique(cc_labels[np.isin(cc_cut_idx, cc_centers)])
            prop_cent_comp = cc_labels[np.where(cc_cut_idx == prop_cent)[0][0]]
            
            if prop_cent_comp in center_comp:
                if peaked[prop_cent] == min(peaked[cc_centers]):
                    cc_centers.append(prop_cent)
                    not_tested[prop_cent] = False
                    continue
                else:
                    break  
            else:
                cc_centers.append(prop_cent)
                not_tested[prop_cent] = False

        peaks.extend(cc_idx[cc_centers])
        
        # Construct BBTree and cluster matrix
        BBTree = np.column_stack((np.arange(nc), cc_big_brother))
        BBTree[cc_centers, 1] = cc_centers
        Clustmat = scipy.sparse.csr_matrix((np.ones(nc), (BBTree[:, 0], BBTree[:, 1])), shape=(nc, nc))
        
        n_clusts, cc_y_pred = scipy.sparse.csgraph.connected_components(Clustmat, directed=True, return_labels=True)
        cc_y_pred += n_cent
        n_cent += n_clusts
        y_pred[cc_idx] = cc_y_pred
        
    return y_pred
class CPFcluster:
    """
    A class to perform CPF (Connected Components and Density-based) clustering.
    Identifies clusters using connected components and density peaks.
    
    Attributes:
        min_samples (int): Minimum number of neighbors to consider for connectivity.
        rho (float): Density parameter controlling the radius cutoff.
        alpha (float): Parameter for edge-cutoff in cluster detection.
        n_jobs (int): Number of parallel jobs for computation.
        remove_duplicates (bool): Whether to remove duplicate data points before clustering.
        cutoff (int): Threshold for filtering out small connected components as outliers.
        distance_metric (str): Metric to use for distance computation.
        merge (bool): Whether to merge similar clusters based on a threshold.
        merge_threshold (float): Distance threshold for merging clusters.
        density_ratio_threshold (float): Density ratio threshold for merging clusters.
        plot_umap (bool): Whether to plot UMAP visualization after clustering.
        plot_pca (bool): Whether to plot PCA visualization after clustering.
    """
    def __init__(self, min_samples=5, rho=0.4, alpha=1, n_jobs=1, remove_duplicates=False, cutoff=1,
                 distance_metric='euclidean', merge=False, merge_threshold=0.5, density_ratio_threshold=0.1,
                 plot_umap=False, plot_pca=False, plot_tsne=False):
        self.min_samples = min_samples
        self.rho = rho
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.remove_duplicates = remove_duplicates
        self.cutoff = cutoff
        self.distance_metric = distance_metric
        self.merge = merge
        self.merge_threshold = merge_threshold
        self.density_ratio_threshold = density_ratio_threshold
        self.plot_umap = plot_umap
        self.plot_pca = plot_pca
        self.plot_tsne = plot_tsne

    def fit(self, X):
        """
        Fits the CPF clustering model to the input data.
        
        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).
        
        Returns:
            None: Updates class attributes with fitted cluster labels.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be an n x d numpy array.")
        if self.remove_duplicates:
            X = np.unique(X, axis=0)
        n, d = X.shape
        if self.min_samples > n:
            raise ValueError("min_samples cannot be larger than n.")

        self.components, self.CCmat, knn_radius = build_CCgraph(X, self.min_samples, self.cutoff, self.n_jobs)
        best_distance, big_brother = get_density_dists_bb(X, self.min_samples, self.components, knn_radius, self.n_jobs)
        self.labels = get_y(self.CCmat, self.components, knn_radius, best_distance, big_brother, self.rho, self.alpha, d)

        if self.merge:
            centroids, densities = self.calculate_centroids_and_densities(X, self.labels)
            self.labels = self.merge_clusters(X, centroids, densities, self.labels)

        self.fitted_ = True
        if self.plot_umap:
            plot_clusters_umap(X, self.labels)
        if self.plot_pca:
            plot_clusters_pca(X, self.labels)
        if self.plot_tsne:
            plot_clusters_tsne(X, self.labels)

    def calculate_centroids_and_densities(self, X, labels):
        """
        Calculates the centroids and average densities of clusters.
        
        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            labels (np.ndarray): Cluster labels for each sample.
        
        Returns:
            centroids (np.ndarray): Centroids of each cluster.
            densities (np.ndarray): Average density of each cluster.
        """
        # Filter out outliers from the labels
        valid_indices = labels != -1
        unique_labels = np.unique(labels[valid_indices])
        centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
        densities = np.array([np.mean(self.CCmat[labels == k, :][:, labels == k]) for k in unique_labels])
        return centroids, densities


    def merge_clusters(self, X, centroids, densities, labels):
        """
        Merges similar clusters based on distance and density ratio thresholds.
        
        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            centroids (np.ndarray): Centroids of each cluster.
            densities (np.ndarray): Average density of each cluster.
            labels (np.ndarray): Cluster labels for each sample.
        
        Returns:
            labels (np.ndarray): Updated cluster labels after merging.
        """
        # Continue using valid_indices to filter labels
        valid_indices = labels != -1 
        unique_labels = np.unique(labels[valid_indices])
        n_clusters = len(centroids)
        merge_map = {}
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if np.linalg.norm(centroids[i] - centroids[j]) < self.merge_threshold and \
                abs(densities[i] - densities[j]) / (max(densities[i], densities[j]) )< self.density_ratio_threshold:
                    smaller = i if densities[i] < densities[j] else j
                    larger = j if smaller == i else i
                    merge_map[smaller] = larger
                    centroids[larger] = (centroids[larger] * np.sum(labels == larger) + centroids[smaller] * np.sum(labels == smaller)) / (np.sum(labels == larger) + np.sum(labels == smaller))
                    densities[larger] = (densities[larger] + densities[smaller]) / 2
        for old, new in merge_map.items():
            labels[labels == old] = new
        # Update labels while excluding outliers
        new_labels = np.array([np.min(np.where(unique_labels == label)[0]) for label in labels if label != -1])
        labels[valid_indices] = new_labels  
        return labels