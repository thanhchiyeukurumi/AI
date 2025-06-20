# spam_detector_ai/classifiers/random_forest_classifier.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from .base_classifier import BaseClassifier


class CustomDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        prob_sq = (counts / len(y)) ** 2
        return 1 - np.sum(prob_sq)

    def _best_split(self, X, y):
        best_gini = float("inf")
        best_idx, best_thresh = None, None

        for idx in range(X.shape[1]):
            thresholds = np.unique(X[:, idx])
            for thresh in thresholds:
                left_mask = X[:, idx] <= thresh
                right_mask = ~left_mask
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                gini = (len(y[left_mask]) * gini_left + len(y[right_mask]) * gini_right) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thresh = thresh

        return best_idx, best_thresh

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return {'leaf': True, 'class': np.bincount(y).argmax()}

        idx, thresh = self._best_split(X, y)
        if idx is None:
            return {'leaf': True, 'class': np.bincount(y).argmax()}

        left_mask = X[:, idx] <= thresh
        right_mask = ~left_mask

        return {
            'leaf': False,
            'feature': idx,
            'threshold': thresh,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_single(self, x, node):
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])


class CustomRandomForest:
    def __init__(self, n_estimators=10, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = CustomDecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)


class RandomForestSpamClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)
        self.smote = SMOTE(random_state=42)

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train).toarray()
        X_train_res, y_train_res = self.smote.fit_resample(X_train_vectorized, y_train)
        self.classifier = CustomRandomForest(n_estimators=10, max_depth=10)
        self.classifier.fit(np.array(X_train_res), np.array(y_train_res))
