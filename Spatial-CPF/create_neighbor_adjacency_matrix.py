import numpy as np
from sklearn.neighbors import NearestNeighbors

def create_neighbor_adjacency_matrix(coords, n_neighbors):
    """
    Create a 0/1 adjacency matrix where 1 indicates that a point is among
    the k-nearest neighbors of another point.
    
    Args:
        coords: numpy array of shape (n_samples, 2) with latitude/longitude in degrees
        n_neighbors: number of nearest neighbors to consider
        
    Returns:
        adjacency_matrix: binary numpy array of shape (n_samples, n_samples)
    """
    # Convert coordinates to radians for haversine metric
    coords_rad = np.radians(coords)
    
    # Create NearestNeighbors model with haversine distance
    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,  # +1 because each point is its own nearest neighbor
        metric='haversine',
        radius=6371  # Earth radius in km (only affects radius_neighbors)
    )
    
    # Fit the model
    nn.fit(coords_rad)
    
    # Find nearest neighbors for all points
    distances, indices = nn.kneighbors(coords_rad)
    
    # Create empty adjacency matrix
    n_samples = coords.shape[0]
    adjacency_matrix = np.zeros((n_samples, n_samples), dtype=int)
    
    # Fill the adjacency matrix
    for i in range(n_samples):
        # Skip the first neighbor (point itself)
        neighbors = indices[i, 1:]  # exclude self
        adjacency_matrix[i, neighbors] = 1
    
    # Make the matrix symmetric (if j is neighbor of i, then i is neighbor of j)
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
    
    return adjacency_matrix


Data = np.loadtxt('Data/G5_shallow_topsoil_WGS84.csv', delimiter=',', skiprows=1)
coordinates = Data[:, :2]
adj_matrix = create_neighbor_adjacency_matrix(coordinates, n_neighbors=75)
np.save('geo_neighbor_adjacency_matrix.npy', adj_matrix)