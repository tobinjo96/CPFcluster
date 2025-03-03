
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from core import CPFcluster
# Note that faiss-gpu is only supported on Linux, not on Windows.
import time
start_time = time.time()


# Define the main function to utilize Python's multiprocessing unit (in Windows OS).
def main():
    # Load the dataset.
    Data = np.load("Data/ecoli.npy")
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
print("--- %s seconds ---" % (time.time() - start_time))
