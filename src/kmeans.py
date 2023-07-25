import numpy as np
import pandas as pd

class KMeans:
    
    def __init__(self, n_clusters=5, max_iter=300):
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self.labels = None

    def fit(self, X):
        centroids = self._initialize_centroids(X)
        itr = 0
        zero_diff_itr = 0

        while itr < self._max_iter and zero_diff_itr <= 10:
            for label in sorted(np.unique(self.labels)):
                indices = np.where(self.labels == label)
                centroids[int(label)] = X.loc[indices].mean()
            diff = self._label_data(X, centroids)
            itr += 1
            if diff == 0:
                zero_diff_itr += 1
        
    def _initialize_centroids(self, X):
        if self._n_clusters == 1:
            self.labels = np.full(len(X), 0)
            return
        else:
            self.labels = np.full(len(X), np.inf)
        idx_centroid = 0
        centroids = list(np.array(X.sample(n=1, axis=0, random_state=2)))
        max_dist = -1
        next_centroid = None
        for index, row in X.iterrows():
            dist = np.linalg.norm(row - centroids[0])
            if dist > max_dist:
                next_centroid = row
                max_dist = dist
        
        centroids.append(np.array(next_centroid))

        self._label_data(X, centroids)
        
        while idx_centroid < self._n_clusters - 2:
            
            for index, row in X.iterrows():
                max_dist_to_centroid = -np.inf
                next_centroid = None
                dist_to_centroid = np.linalg.norm(centroids[idx_centroid] - row)
                if max_dist_to_centroid < dist_to_centroid:
                    max_dist_to_centroid = dist_to_centroid
                    next_centroid = row
            
            idx_centroid += 1
            centroids.append(np.array(next_centroid))

            self._label_data(X, centroids)
        
        return centroids
    
    def _label_data(self, X, centroids):
        diff = 0
        for index, row in X.iterrows():
            min_dist_to_centroid = np.inf
            assign_centroid = -1
            for label, centroid in enumerate(centroids):
                dist_to_centroid = np.linalg.norm(centroid - row)
                if dist_to_centroid < min_dist_to_centroid:
                    assign_centroid = label
                    min_dist_to_centroid = dist_to_centroid
            if self.labels[index] != assign_centroid:
                diff += 1
            self.labels[index] = assign_centroid
        return diff