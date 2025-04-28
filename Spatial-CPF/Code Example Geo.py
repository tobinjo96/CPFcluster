
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
from core_Geo import CPFcluster
# Note that faiss-gpu is only supported on Linux, not on Windows.
import time
start_time = time.time()


# Define the main function to utilize Python's multiprocessing unit (in Windows OS).
def main():
    # Load the dataset.
    
    Data = np.loadtxt('Data/G5_shallow_topsoil_WGS84.csv', delimiter=',', skiprows=1)
    X = Data[:, 2:]

    # Normalize dataset for easier hyperparameter tuning.
    X = StandardScaler().fit_transform(X)

    # Initialize CPFcluster with multiple rho, alpha, merge_threshold and density_ratio_threshold values.
    cpf = CPFcluster(
        min_samples=75,
        rho=[0.01, 0.02, 0.03, 0.04],  # list of rho values for grid search
        alpha=[0.015, 0.03, 0.045, 0.06],  # list of alpha values for grid search
        merge=True,
        merge_threshold=[7, 7.5, 8, 8.5],  # list of merge thresholds
        density_ratio_threshold=[0.6, 0.7, 0.8, 0.9],  # list of density ratio thresholds
        n_jobs=-1,
        plot_tsne=True,
        plot_pca=True,
        plot_umap=True
    )

    # Fit the model for a range of min_samples values.
    print("Fitting CPFcluster...")
    geo_neighbor_adjacency_matrix = np.load("Data/geo_neighbor_adjacency_matrix.npy")
    cpf.fit(X, geo_neighbor_adjacency_matrix, k_values=[65, 70, 75])

    # Perform cross-validation to find the best (min_samples, rho, alpha, merge_threshold, density_ratio_threshold)
    print("Performing cross-validation...")
    best_params, best_score = cpf.cross_validate(X, validation_index=calinski_harabasz_score)
    print(f"Best Parameters: min_samples={best_params[0]}, rho={best_params[1]:.4f}, alpha={best_params[2]:.4f}, "
        f"merge_threshold={best_params[3]:.4f}, density_ratio_threshold={best_params[4]:.4f}. "
        f"Best Validation Score (Calinski-Harabasz Index): {best_score:.4f}")


    # Access the cluster labels for the best paramter configuration.
    best_labels = cpf.clusterings[best_params]
    print("Cluster labels for best parameters:", best_labels)
    np.savetxt('data_labels.csv', best_labels, delimiter=',', fmt='%f')


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
print("--- %s seconds ---" % (time.time() - start_time))
