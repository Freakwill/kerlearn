#!/usr/bin/env python

"""
General Kernel learing
- local models
- lazy learning
"""


import numpy as np

from sklearn.base import TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from utils import normalize


def inner(func):
    # work with inner sample
    def mthd(obj, X=None, *args, **kwargs):
        if X is None:
            X = obj.X_train_
        result = func(obj, X, *args, **kwargs)
        return result
    return mthd


class KernelMixin:

    @inner
    def get_kernel(self, X, X_=None):
        # compute kernel matrix (normalized) if it dose not have `kernel_matrix_`

        if not hasattr(self, 'kernel_matrix_') or self.kernel_matrix_ is None or X_ is not None:
            if X_ is None:
                X_ = self.X_train_
            self.kernel_matrix_ = self._get_kernel(X, X_)
        return self.kernel_matrix_

    def _get_kernel(self, X, X_=None):
        # compute kernel matrix (normalized)
        if X_ is None:
            X_ = self.X_train_
        return self.kernel(X, X_)

    def fit_kernel(self, X, y=None):
        self.fit(X, y)
        if not hasattr(self, 'kernel_matrix_') or self.kernel_matrix_ is None:
            self.kernel_matrix_ = self._get_kernel(X)
        return self

    def kernel_fit(self, y=None):
        return self


class SimilarityMixin(KernelMixin):

    def get_similarity(self, X, X_=None):
        # compute similarity matrix (normalized) if it dose not have `similarity_matrix_`
        if not hasattr(self, 'similarity_matrix_') or self.similarity_matrix_ is None:
            if X_ is None:
                X_ = self.X_train_
            self.similarity_matrix_ = self._get_similarity(X, X_)
        return self.similarity_matrix_

    def _get_similarity(self, X, X_=None):
        # compute similarity matrix (normalized)
        if X_ is None:
            X_ = self.X_train_
        return self.similarity(X, X_)

    def fit_similarity(self, X, y=None):
        self.fit(X, y)
        if not hasattr(self, 'similarity_matrix_') or self.similarity_matrix_ is None:
            self.similarity_matrix_ = self._get_similarity(X)
        return self


class LocalModelMixin(KernelMixin, TransformerMixin):

    def predict(self, X, y_=None):
        # predicting
        K = self._get_kernel(X, self.X_train_)
        return self.kernel_predict(K, self.y_train_)

    def inner_predict(self, X=None, y=None):
        """Inner-sample predicting
        
        Args:
            X (2D array): input sample/design matrix
            y (None, optional): output
        
        Returns:
            The prediction result of X
        """
        K = self.get_kernel(X, X_=None)
        return self.kernel_predict(K, y)

    def kernel_predict(self, K, y=None):
        # predict with kernel directly
        if y is None:
            y = self.y_train_
        return K @ y

    def kernel_score(self, K, y=None, y_=None):
        # predict with kernel directly
        y_pred = self.kernel_predict(K, y_)
        return r2_score(y, y_pred)

    def _get_kernel(self, X, X_=None):
        # compute kernel matrix (normalized)
        if X_ is None:
            X_ = getattr(self, 'X_train_', X)
        return normalize(self.kernel(X, X_), lb=0.0001)


class LazyMapMixin(TransformerMixin):

    # def inner_predict(self, X, y=None):
    #     # inner-sample predicting
    #     # use self.get_kernel(X, X_=None), instead of _get_kernel
    #     raise NotImplementedError

    def inner_transform(self, *args, **kwargs):
        # lazy transform
        return self.inner_predict(*args, **kwargs)

    def transform(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    reconstruct = transform

    def init_transform(self, X):
        # initial value of the output of X
        if hasattr(self, 'init_predict'):
            return self.init_predict(X)
        else:
            pca = PCA(n_components=self.n_components)
            return pca.fit_transform(X)

    # def inner_predict(self, X):
    #     return self.predict(X)

    def fit(self, X=None):
        """Predict iteration, the core method of the class
        
        Args:
            X (2D array): input sample/design matrix
        
        Returns:
            LazyMapMixin
        """

        self._init(X)
        self.y_train_ = self.init_predict(X)
        for _ in range(self.max_iter):
            y = self.y_train_
            self._supervised_fit(X, y)
            self.y_train_ = self.inner_predict(X)
            if self.control(y, self.y_train_):
                break
        return self

    def _init(self, X, y=None):
        self.X_train_ = X
        self.y_train_ = y
        self.n_samples_ = len(X)

    def _supervised_fit(self, X, y):
        self.y_train_ = y
        return self

    def control(self, y, y_):
        return np.allclose(y, y_)


class SelfLocalModelMixin:

    def predict(self, X, max_iter=5):
        y = self.init_predict(X)
        for _ in range(max_iter):
            K = self._get_kernel(X, X_=self.X_train_, y=y, y_=self.y_train_)
            y = self.kernel_predict(K)
        return y

    max_predict_iter = 10

    def predict(self, X, max_iter=None):
        # predicting for the test sample
        y = self.init_predict(X)
        for _ in range(max_iter or self.max_predict_iter):
            L = self._get_kernel(X=X, y=y)
            y_ = self.kernel_predict(L)
            if np.allclose(y, y_):
                break
            else:
                y = y_
        return y

    def init_predict(self, X):
        K = self._get_kernel(X, X_=self.X_train_)
        y = self.kernel_predict(K)
        return y

    def get_kernel(self, X, X_=None, y=None, y_=None):
        # compute kernel matrix (normalized) if it dose not have `kernel_matrix_`
        if not hasattr(self, 'kernel_matrix_') or self.kernel_matrix_ is None:
            if X_ is None:
                X_ = getattr(self, 'X_train_', X)
            if y_ is None:
                y_ = getattr(self, 'X_train_', y)
            self.kernel_matrix_ = self._get_kernel(X, X_, y, y_)
            self.y_kernel_matrix_ = self.y_kernel(y, y_)
            self.x_kernel_matrix_ = self.kernel(X, X_)
        return self.kernel_matrix_


class AutoencoderMixin(TransformerMixin):

    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))

    def score(self, X):
        return super().score(X, X)


class SelfSupervisedMixin(AutoencoderMixin):

    def fit(self, X):
        return self.model_cls.fit(self, X, X)

    def partial_fit(self, X):
        return self.model_cls.partial_fit(self, X, X)


class LocalClassifierMixin(LocalModelMixin, ClassifierMixin):

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return self

class LocalRegressorMixin(LocalModelMixin, RegressorMixin):
    # Local regression

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        return self