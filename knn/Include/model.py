import numpy as np
from collections import Counter
from enum import Enum
from sklearn.preprocessing import StandardScaler

class Distances(Enum):
    EUCLID = 'euclid'
    MANHATTAN = 'manhattan'

class KNNCustom:
    def __init__(self, k=3, distance_m='euclid'):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.distance_m = Distances(distance_m.lower())
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _distance(self, x1, x2):
        if self.distance_m == Distances.EUCLID:
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_m == Distances.MANHATTAN:
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_m}")

    def _predict_outlier(self, x, true_label):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        majority_label = Counter(k_labels).most_common(1)[0][0]
        return 0 if majority_label == true_label else 1

    def predict(self, X, y_true=None, detect_outliers=False):
        X = self.scaler.transform(X)
        if detect_outliers:
            return np.array([self._predict_outlier(x, y) for x, y in zip(X, y_true)])
        else:
            return np.array([self._predict_class(x) for x in X])

    def _predict_class(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_labels).most_common(1)[0][0]
