import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        """
        Parameters:
        - n_clusters (int): Jumlah cluster yang ingin dibentuk.
        - max_iters (int): Jumlah iterasi maksimum.
        - random_state (int): Seed untuk reproduktifitas inisialisasi.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.X = None 

    def _euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    def fit(self, X):
        """
        Parameters:
        - X (array-like): Data training.
        """
        self.X = np.array(X) 
        n_samples, n_features = self.X.shape

        # Inisialisasi centroids secara acak
        np.random.seed(self.random_state)
        random_sample_idxs = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = self.X[random_sample_idxs]

        # Iterasi untuk optimisasi centroids
        for _ in range(self.max_iters):
            clusters = self._create_clusters(self.X)
            old_centroids = self.centroids
            new_centroids = self._calculate_new_centroids(clusters, n_features)
            if self._is_converged(old_centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        """
        Parameters:
        - X (array-like): Data yang ingin diprediksi.
        """
        X_np = np.array(X)
        clusters = self._create_clusters(X_np)
        labels = np.empty(X_np.shape[0])
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, X_np):
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, sample in enumerate(X_np):
            centroid_idx = self._closest_centroid(sample)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample):
        distances = [self._euclidean_distance(sample, point) for point in self.centroids]
        return np.argmin(distances)

    def _calculate_new_centroids(self, clusters, n_features):
        new_centroids = np.zeros((self.n_clusters, n_features))
        for i, cluster in enumerate(clusters):
            if cluster: 
                new_mean = np.mean(self.X[cluster], axis=0) 
                new_centroids[i] = new_mean
        return new_centroids

    def _is_converged(self, old_centroids, new_centroids):
        distances = [self._euclidean_distance(old_centroids[i], new_centroids[i]) for i in range(self.n_clusters)]
        return sum(distances) == 0

