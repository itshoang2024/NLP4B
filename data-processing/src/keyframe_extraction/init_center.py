import numpy as np
from sklearn.cluster import KMeans

def kmeans_init(data):
    print("In the process of initialising the center")
    n = len(data)
    sqrt_n = int(np.sqrt(n))
    
    # We cannot create more clusters than the number of unique points
    unique_data = np.unique(data, axis=0)
    k_init = min(sqrt_n, len(unique_data))
    
    if k_init <= 1:
        # Degenerate case: all points are identical or empty
        labels = np.zeros(n, dtype=int)
        centers = [data[0]] if n > 0 else []
        return labels, np.array(centers)
        
    # Use sklearn KMeans for extremely fast initialization compared to original O(N^3) logic
    kmeans = KMeans(n_clusters=k_init, init='k-means++', n_init=1, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # The original algorithm expected the centers to be exact data points
    centers = []
    for i in range(k_init):
        cluster_samples = data[labels == i]
        if len(cluster_samples) == 0:
            continue
        cluster_mean = kmeans.cluster_centers_[i]
        # Find the actual point closest to the cluster mean
        distances = np.linalg.norm(cluster_samples - cluster_mean, axis=1)
        centers.append(cluster_samples[np.argmin(distances)])
    
    return labels, np.array(centers)
