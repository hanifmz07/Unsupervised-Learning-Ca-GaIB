import numpy as np
import pandas as pd

class KMedoids:
    
    def __init__(self, n_clusters=5, max_iter=100):
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self.medoids = None
        self.medoids_cost = None
        self.labels = None

    def fit(self, X):
        self.medoids = self._initialize_medoids(X)
        itr = 0
        old_medoids = self.medoids.copy()
        new_medoids = self.medoids.copy()
        new_labels = self.labels.copy()
        new_cost = np.inf
        while itr < self._max_iter:
            for label in sorted(np.unique(self.labels)):
                for index, row in X.iterrows():
                    new_medoids[label] = row
                    new_labels = self._label_data(X, new_medoids)
                    new_cost = self._count_cost(X, new_labels, new_medoids)
                    if new_cost < self.medoids_cost:
                        self.medoids_cost = new_cost
                        self.medoids = new_medoids.copy()
                        self.labels = new_labels.copy()
            condition_medoid = np.array(old_medoids) == np.array(self.medoids)
            if condition_medoid.all():
                break
            old_medoids = self.medoids.copy()
            itr += 1

    def _initialize_medoids(self, X):
        self.labels = np.full(len(X), 0)
        min_total_dist = np.inf
        min_medoid = None
        index_min_medoid = None
        medoids = []
        medoids_indices = []
        for index, row in X.iterrows():
            total_dist = 0
            for index2, row2 in X.iterrows():
                total_dist += self._manhattan_distance(row, row2)
            if total_dist <= min_total_dist:
                min_total_dist = total_dist
                min_medoid = row
                index_min_medoid = index
        medoids.append(min_medoid)
        medoids_indices.append(index_min_medoid)

        for i in range(1, self._n_clusters):
            min_medoid = None
            index_min_medoid = None
            check_X = X.drop(index=medoids_indices)
            min_cost = np.inf
            min_labels = None
            for index, row in check_X.iterrows():
                check_medoids = medoids + [row]
                check_labels = self._label_data(X, check_medoids)
                cost = self._count_cost(X, check_labels, check_medoids)
                if min_cost > cost:
                    min_cost = cost
                    min_labels = check_labels
                    min_medoid = row
                    index_min_medoid = index
            medoids.append(min_medoid)
            medoids_indices.append(index_min_medoid)
            
            self.medoids_cost = min_cost
            self.labels = min_labels
        
        return medoids

    def _label_data(self, X, medoids):
        new_labels = self.labels.copy()
        for index, row in X.iterrows():
            min_dist_to_medoid = np.inf
            assign_medoid = -1
            for label, medoid in enumerate(medoids):
                dist_to_medoid = self._manhattan_distance(medoid, row)
                if dist_to_medoid < min_dist_to_medoid:
                    assign_medoid = label
                    min_dist_to_medoid = dist_to_medoid
            new_labels[index] = assign_medoid
        return new_labels
    
    def _count_cost(self, X, labels, medoids):
        cost = 0
        for label in sorted(np.unique(labels)):
            indices = np.where(labels == label)
            cost += self._manhattan_distance(np.array(X.loc[indices]), np.array(medoids[label]))
        return cost
    
    def _manhattan_distance(self, point1, point2):
        return np.sum(np.abs(point1 - point2))