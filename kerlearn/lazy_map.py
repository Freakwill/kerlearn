#!/usr/bin/env python

"""
kernel
"""

import numpy as np
import numpy.linalg as LA

from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

from kerlearn import LazyMapMixin, KernelMixin


class KernelLazyMap(LazyMapMixin, KernelMixin):

    def __init__(self, kernel=rbf_kernel, n_components=2):
        self.kernel = kernel
        self.n_components = n_components
        self.lambda_ = 0.01

    def supervised_fit(self, X, y):
        K = self.get_kernel(X)
        self.X_train_ = X
        M = K.shape[0]
        self.alpha_ = LA.lstsq(self.lambda_ * np.eye(M) + K, y, rcond=None)[0]
        # self.alpha_ = slim(self.alpha_, self.threshold)
        return self

    def transform(self, X):
        # X = np.random.randn(X.shape)
        return self.kernel(X, self.X_train_) @ self.alpha_

    def inner_transform(self, X):
        K = self.get_kernel(X)
        M = K.shape[0]
        if hasattr(self, 'alpha_'):
            alpha_ = self.alpha_
        else:
            alpha_ = self.alpha_ = LA.lstsq(self.lambda_ * np.eye(M) + K, self.y_train_, rcond=None)[0]
        return K @ alpha_

    def control(self, y, y_):
        return np.sum(np.abs(y-y_))<0.0001

    def fit(self, X):
        self._init(X)
        self.y_train_ = self.init_transform(X)
        y = self.y_train_
        self._supervised_fit(X, y)
        self.y_train_ = self.inner_transform(X)
        return self


class LinearLazyMap(LazyMapMixin):

    def __init__(self, n_components=2, max_iter=20):
        self.n_components = n_components
        self.max_iter = max_iter
        self.lambda_ = 0.1

    def supervised_fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        self.gram_ = (X.T @ X + self.lambda_)
        self.b_ = X.T @ y
        self.hat_ = X @ LA.lstsq(self.gram_, X.T, rcond=None)[0]
        return self

    def fit(self, X):
        super().fit(X)
        self.b_ = self.X_train_.T @ self.y_train_
        return self

    def _init(self, X):
        self.X_train_ = X
        self.gram_ = (X.T @ X + self.lambda_)
        self.hat_ = X @ LA.lstsq(self.gram_, X.T, rcond=None)[0]
        return self

    def transform(self, X):
        # X = np.random.randn(X.shape)
        return X @ LA.lstsq(self.gram_, self.b_, rcond=None)[0]

    def inner_transform(self, X):
        if hasattr(self, 'hat_'):
            hat_ = self.hat_
        else:
            hat_ = self.hat_ = X @ LA.lstsq((X.T @ X + self.lambda_), X.T, rcond=None)[0]
        return hat_ @ self.y_train_

    def control(self, y, y_):
        return np.sum(np.abs(y-y_))<0.0001

    def init_predict(self, X):
        return self.hat_ @ self.y_train_

