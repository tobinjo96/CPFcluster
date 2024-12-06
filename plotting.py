def plot_clusters_umap(X, labels, sample_size=30000, n_neighbors=15, min_dist=0.3, random_state=42):
    """
    Visualizes clusters using UMAP for dimensionality reduction on provided data X with labels.
    
    Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
        sample_size (int): Maximum number of samples to use for UMAP (for large datasets).
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance parameter for UMAP.
        random_state (int): Random state for reproducibility.
    """
    import umap
    import matplotlib.pyplot as plt
    import numpy as np

    # If dataset is large, sample for visualization
    if X.shape[0] > sample_size:
        subset_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
        X = X[subset_indices]
        labels = labels[subset_indices]

    # Optimize UMAP for speed
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=4,
        low_memory=True,
    )
    X_reduced = reducer.fit_transform(X)

    # Group cluster points efficiently
    unique_labels = np.unique(labels)
    cluster_points = {label: X_reduced[labels == label] for label in unique_labels}

    # Initialize plot
    plt.figure(figsize=(10, 6))

    # Plot clusters
    for label, points in cluster_points.items():
        if label != -1:  # Exclude outliers
            plt.scatter(points[:, 0], points[:, 1], s=10)

    # Plot outliers separately
    if -1 in cluster_points:
        outliers = cluster_points[-1]
        plt.scatter(outliers[:, 0], outliers[:, 1], c="gray", label="Outliers", alpha=0.5, s=10)

    # Finalize plot
    plt.title("Clusters and Outliers (UMAP)")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")

    # Configure legend to show only outliers
    handles, labels = plt.gca().get_legend_handles_labels()
    outlier_handles = [h for h, lbl in zip(handles, labels) if lbl == "Outliers"]
    if outlier_handles:
        plt.legend(outlier_handles, ["Outliers"], loc="lower left", fontsize="small")

    plt.show()

def plot_clusters_pca(X, labels):
    """
    Visualizes the clusters and outliers using PCA on provided data X with labels.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np

    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Initialize plot
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)

    # Plot clusters
    for label in unique_labels:
        cluster_points = X_reduced[labels == label]
        if label == -1:
            # Plot outliers separately
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c="gray", label="Outliers", alpha=0.5, s=10)
        else:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10)  # No label for clusters

    # Add title and axis labels
    plt.title("Clusters and Outliers (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Configure legend to show only outliers
    handles, labels = plt.gca().get_legend_handles_labels()
    outlier_handles = [h for h, lbl in zip(handles, labels) if lbl == "Outliers"]
    if outlier_handles:
        plt.legend(outlier_handles, ["Outliers"], loc="lower left", fontsize="small")

    plt.show()

def plot_clusters_tsne(X, labels, sample_size=30000, perplexity=30, random_state=42):
    """
    Visualizes clusters using t-SNE for dimensionality reduction on provided data X with labels.

    Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
        sample_size (int): Maximum number of samples to use for t-SNE (for large datasets).
        perplexity (float): Perplexity parameter for t-SNE.
        random_state (int): Random state for reproducibility.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    # If dataset is large, sample for visualization
    if X.shape[0] > sample_size:
        subset_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
        X = X[subset_indices]
        labels = labels[subset_indices]

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=4)
    X_reduced = tsne.fit_transform(X)

    # Group cluster points efficiently
    unique_labels = np.unique(labels)
    cluster_points = {label: X_reduced[labels == label] for label in unique_labels}

    # Initialize plot
    plt.figure(figsize=(10, 6))

    # Plot clusters
    for label, points in cluster_points.items():
        if label != -1:  # Exclude outliers
            plt.scatter(points[:, 0], points[:, 1], s=10)  # No label for clusters

    # Plot outliers separately
    if -1 in cluster_points:
        outliers = cluster_points[-1]
        plt.scatter(outliers[:, 0], outliers[:, 1], c="gray", label="Outliers", alpha=0.5, s=10)

    # Add title and axis labels
    plt.title("Clusters and Outliers (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Configure legend to show only outliers
    handles, labels = plt.gca().get_legend_handles_labels()
    outlier_handles = [h for h, lbl in zip(handles, labels) if lbl == "Outliers"]
    if outlier_handles:
        plt.legend(outlier_handles, ["Outliers"], loc="lower left", fontsize="small")

    plt.show()
