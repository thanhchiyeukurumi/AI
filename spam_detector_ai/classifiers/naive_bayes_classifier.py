# spam_detector_ai/classifiers/naive_bayes_classifier.py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from .base_classifier import BaseClassifier


class CustomMultinomialNB:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        class_count = np.zeros(n_classes, dtype=np.float64)
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, label in enumerate(self.classes_):
            X_class = X[y == label]
            class_count[idx] = X_class.shape[0]
            feature_count[idx, :] = X_class.sum(axis=0)

        # Apply Laplace smoothing
        smoothed_fc = feature_count + 1
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)

        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)
        self.class_log_prior_ = np.log(class_count / class_count.sum())

    def predict(self, X):
        X = np.array(X)
        log_probs = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(log_probs, axis=1)]


class NaiveBayesClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.vectoriser = CountVectorizer(**BaseClassifier.VECTORIZER_PARAMS)

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train).toarray()
        self.classifier = CustomMultinomialNB()
        self.classifier.fit(X_train_vectorized, y_train)
