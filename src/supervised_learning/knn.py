import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Parameters:
        - k (int): Jumlah tetangga terdekat yang akan digunakan untuk voting.
        - distance_metric (str): Metrik jarak yang akan digunakan. (Pilihan: 'euclidean', 'manhattan', 'minkowski'.)
        """
        self.k = k
        self.distance_metric = distance_metric
        self.p = 2 

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def _minkowski_distance(self, x1, x2):
        return np.power(np.sum(np.power(np.abs(x1 - x2), self.p)), 1/self.p)

    def fit(self, X_train, y_train):
        """
        Parameters:
        - X_train (array-like): Fitur dari data training.
        - y_train (array-like): Label dari data training.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Parameters:
        - X_test (array-like): Data yang ingin diprediksi.
        """
        X_test_np = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test_np]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        Parameters:
        - x (array): Satu titik data yang ingin diprediksi.
        """
        distances = []
        for x_train_point in np.array(self.X_train):
            if self.distance_metric == 'euclidean':
                dist = self._euclidean_distance(x, x_train_point)
            elif self.distance_metric == 'manhattan':
                dist = self._manhattan_distance(x, x_train_point)
            elif self.distance_metric == 'minkowski':
                dist = self._minkowski_distance(x, x_train_point)
            else:
                raise ValueError("Distance metric not supported.")
            distances.append(dist)

        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train.iloc[i] for i in k_nearest_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
