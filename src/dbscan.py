import pandas as pd
import numpy as np

class DBScan:
    """DBScan algorithm"""
    def __init__(self, eps=0.3, min_samples=5):
        self._eps = eps
        self._min_samples=self._min_samples
        self.labels = None
        self.n_clusters = 0
    
    def fit(self, X):
        # Initialize labels with np.inf to denote unlabeled points
        self.labels = np.full(len(X), np.inf)
        for index, point in X.iterrows():
            if self.labels[index] != np.inf:
                continue
            neighbours = self._search_neighbors(X, point)
            if len(neighbours) < self._min_samples:
                self.labels[index] = -1
                continue
            self.n_clusters += 1
            self.labels[index] = self.n_clusters
            while neighbours:
                index2 = next(iter(neighbours))
                if self.labels[index2] == -1:
                    self.labels[index2] = self.n_clusters
                if self.labels[index2] != np.inf:
                    neighbours.pop(index2)
                    continue
                self.labels[index2] = self.n_clusters
                neighbours2 = self._search_neighbors(X, neighbours[index2])
                if len(neighbours2) >= self._min_samples:
                    neighbours.update(neighbours2)
                neighbours.pop(index2)

    def _search_neighbors(self, X, point):
        neighbours = {}
        for index, row in X.iterrows():
            if np.linalg.norm(point - row) <= self._eps and point.name != index:
                neighbours[index] = row
        return neighbours