from sklearn.datasets import make_moons, make_blobs, make_circles

def get_dataset(name="moons", n_samples=1000, noise=0.1, n_features=2, centers=None):
    """
    Generates X and y for NN testing.
    
    Args:
        name (str): 'moons', 'circles', or 'blobs'
        n_samples (int): Number of data points
        noise (float): Standard deviation of Gaussian noise
        
    Returns:
        X (np.array): Input data of shape (n_samples, 2)
        y (np.array): Labels of shape (n_samples, 1)
    """
    
    if name == "moons":
        # Great for testing non-linearity (ReLU/Tanh/Sigmoid)
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    
    elif name == "circles":
        # Harder non-linear problem (concentric circles)
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        
    elif name == "blobs":
        # Simple linear problem (good for testing basic weights/biases update)
        X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=42)
        
    else:
        raise ValueError("Invalid dataset name. Choose 'moons', 'circles', or 'blobs'.")

    y = y.flatten()
    
    return X, y