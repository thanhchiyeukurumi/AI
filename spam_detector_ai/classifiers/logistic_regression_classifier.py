# spam_detector_ai/classifiers/logistic_regression_classifier.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_classifier import BaseClassifier


class CustomLogisticRegression:
    def __init__(self, lr=0.1, max_iter=200, C=10):
        self.lr = lr
        self.max_iter = max_iter
        self.C = C  # Inverse of regularization strength
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y = np.array(y)

        for _ in range(self.max_iter):
            linear_model = X.dot(self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Gradient computation
            dw = (1 / n_samples) * X.T.dot(y_predicted - y) + (1 / self.C) * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = X.dot(self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)


class LogisticRegressionSpamClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train).toarray()
        self.classifier = CustomLogisticRegression(lr=0.1, max_iter=200, C=10)
        self.classifier.fit(X_train_vectorized, y_train)
