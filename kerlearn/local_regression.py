#!/usr/bin/env python

"""Local Regression/Mean

y^(X*) ~ K(X*,X)y / K(X*,X)1
"""

import numpy as np
import numpy.linalg as LA

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import *

from base import LocalRegressorMixin, SelfLocalModelMixin

from kernels import *

from utils import normalize

def normal_hollow(K):
    np.fill_diagonal(K, 0)
    K = normalize(K, lb=0.0001)
    np.fill_diagonal(K, -1)
    return K


_lb = 0.00001


def add_const(X):
    N = len(X)
    return np.column_stack((X, np.ones(N)))


def to_column_vector(arr):
    return np.atleast_2d(arr).T if arr.ndim == 1 else arr


class LocalRegression(LocalRegressorMixin, BaseEstimator):
    # Local regression

    def __init__(self, kernel=rbf_kernel, lambda_=0.0001, **kwargs):
        self.kernel = kernel
        self.correct_rate = 0.1
        self.lambda_ = lambda_

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        return self

    def inner_predict(self, y=None):
        # predicting for the train sample
        X = self.X_train_
        if y is None:
            y = self.y_train_
        L = self.get_kernel(X)
        return self.kernel_predict(L, y)

    def predict(self, X):
        # predicting for the test sample
        L = self._get_kernel(X)
        return self.kernel_predict(L)

    def loo_score(self, X, y):
        self.fit(X, y)
        L = self._get_kernel(X, X)
        np.fill_diagonal(L, 0)
        return r2_score(y, L @ y)

    def correct_shift(self, X=None, y=None, max_iter=5):
        if y is None:
            y = self.y_train_
        if X is None:
            X = self.X_train_
        yc = y.copy()
        for _ in range(max_iter):
            y = self.predict(X)
            y += self.correct_rate * (yc - y)
        return y


class LocalMeanRegression(LocalRegression):
    # Local Mean regression

    def _get_normal_kernel(self, lb=0.0001, *args, **kwargs):
        return normalize(self._get_kernel(*args, **kwargs), lb=lb)


class LocalLinearRegression(LocalRegression):
    # Local liear regression

    def __init__(self, fit_intercept=True, **kwargs):
        super().__init__(**kwargs)
        self.fit_intercept = fit_intercept

    def _get_kernel(self, X, X_=None, y=None, y_=None):
        # calculate the kernel matrix (normalized)
        if X_ is None:
            X_ = getattr(self, 'X_train_', X)
        if y_ is None:
            y_ = getattr(self, 'y_train_', y)
        N, _ = X.shape
        Ntrain, _ = X_.shape
        if self.fit_intercept:
            X_ = add_const(X_)
            X = add_const(X)
            K = self.kernel(X, X_)
        L = np.empty((N, Ntrain))
        for i, (x, k) in enumerate(zip(X, K)):
            XTD = X_.T * k
            M = XTD @ X_
            L[i] = x @ LA.lstsq(XTD @ X_ + self.lambda_ * np.eye(M.shape[0]), XTD, rcond=None)[0]
        return L


class SelfLocalRegression(SelfLocalModelMixin, LocalMeanRegression):
    # Local Mean regression

    def init_predict(self, X):
        L = self._get_kernel(X=X)
        return self.kernel_predict(L)

    def inner_predict(self, y=None):
        X = self.X_train_
        if y is None:
            y = self.y_train_
        L = self.get_kernel(X=X, y=y)
        L += self.lambda_ * np.eye(L.shape[0])
        return self.kernel_predict(L, y) 


class SelfLocalMeanRegression(SelfLocalRegression, LocalMeanRegression):
    # Local liear regression

    def __init__(self, kernel=rbf_kernel, y_kernel=rbf_kernel, **kwargs):
        super().__init__(kernel, **kwargs)
        self.y_kernel = y_kernel

    def _get_kernel(self, X, X_=None, y=None, y_=None):
        # calculate the kernel matrix of x and y(normalized)

        Kx  = self._get_x_kernel(X, X_)

        if y is None or self.y_kernel is None:
            return normalize(Kx)
        else:
            return normalize(Kx * self._get_y_kernel(y, y_), lb=_lb)

    def _get_x_kernel(self, X, X_=None):
        # calculate the kernel matrix of x(unnormalized)
        if X_ is None:
            X_ = getattr(self, 'X_train_', X)
        return self.kernel(X, X_)

    def _get_y_kernel(self, y, y_=None):
        # calculate the kernel matrix of y(unnormalized)
        if y_ is None:
            y_ = getattr(self, 'y_train_', y)
        return self.y_kernel(to_column_vector(y), to_column_vector(y_))


from kernels import epanechnikov_kernel, make_kernel

class EpanechnikovRegression(LocalMeanRegression):

    def __init__(self, width=1, **kwargs):
        kernel = make_kernel(epanechnikov_kernel, gamma=1/width**2)
        super().__init__(kernel, **kwargs)


class GaussianRegression(LocalMeanRegression):

    def __init__(self, width=1, **kwargs):
        kernel = make_kernel(rbf_kernel, gamma=1/(2*width**2))
        super().__init__(kernel, **kwargs)


class NeighborRegression(LocalMeanRegression):
    # Local Mean regression

    def __init__(self, width=1, **kwargs):
        kernel = make_kernel(neighbor_kernel, width=width)
        super().__init__(kernel, **kwargs)


class KNNRegression(LocalMeanRegression):
    # Local Mean regression

    def __init__(self, n_neighbors=10, **kwargs):
        kernel = make_kernel(knn_kernel, n_neighors=n_neighbors)
        super().__init__(kernel, **kwargs)


class SelfEpanechnikovRegression(SelfLocalMeanRegression):

    def __init__(self, width=1, y_width=1, **kwargs):
        self._width = width
        self._y_width = y_width
        kernel = make_kernel(epanechnikov_kernel, gamma=1/width**2)
        y_kernel = make_kernel(epanechnikov_kernel, gamma=1/y_width**2)
        super().__init__(kernel, y_kernel, **kwargs)

    @property
    def width(self):
        return self._width

    @property
    def y_width(self):
        return self._y_width

    @width.setter
    def width(self, v):
        self._width = v
        self.kernel = make_kernel(epanechnikov_kernel, gamma=1/v**2)

    @y_width.setter
    def y_width(self, v):
        self._y_width = v
        self.y_kernel = make_kernel(epanechnikov_kernel, gamma=1/v**2)


class SelfGaussianRegression(SelfLocalMeanRegression):

    def __init__(self, width=1, y_width=1, **kwargs):
        self._width = width
        self._y_width = y_width
        kernel = make_kernel(rbf_kernel, gamma=1/width**2)
        y_kernel = make_kernel(rbf_kernel, gamma=1/y_width**2)
        super().__init__(kernel, y_kernel, **kwargs)

    @property
    def width(self):
        return self._width

    @property
    def y_width(self):
        return self._y_width

    @width.setter
    def width(self, v):
        self._width = v
        self.kernel = make_kernel(rbf_kernel, gamma=1/v**2)

    @y_width.setter
    def y_width(self, v):
        self._y_width = v
        self.y_kernel = make_kernel(rbf_kernel, gamma=1/v**2)


class SelfLocalLinearRegression(SelfLocalModelMixin, LocalRegression):
    # Local liear regression

    def __init__(self, kernel=rbf_kernel, y_kernel=rbf_kernel, **kwargs):
        super().__init__(kernel, **kwargs)
        self.fit_intercept = True
        self.lambda_ = 0.01
        self.y_kernel = y_kernel

    def _get_kernel(self, X, X_=None, y=None, y_=None):
        if X_ is None:
            X_ = self.X_train_
        if y_ is None:
            y_ = self.y_train_
        N, _ = X.shape
        Ntrain, _ = X_.shape
        if self.fit_intercept:
            X_ = add_const(X_)
            X = add_const(X)
        if y is None:
            K1 = normalize(self.kernel(X, X_), lb=_lb)
            y = self.kernel_predict(K1)

        K = self.kernel(X, X_) * self.y_kernel(to_column_vector(y), to_column_vector(y_))
        L = np.empty((N, Ntrain))
        for i, (x, k) in enumerate(zip(X, K)):
            XTD = X_.T * k
            L[i] = x @ LA.lstsq(XTD @ X_ + self.lambda_, XTD, rcond=None)[0]
        return L


class NonLocalMeanRegression(SelfLocalMeanRegression):

    delta = 10
    nonlocal_size = 10

    def _get_kernel(self, X, X_=None, y=None, y_=None):
        # calculate the kernel matrix (normalized)
        if X_ is None:
            X_ = getattr(self, 'X_train_', X)
        if y_ is None:
            y_ = getattr(self, 'y_train_', y)
        N, _ = X.shape

        D = euclidean_distances(X)

        z = np.empty((N, self.nonlocal_size))
        for i, (xi, yi) in enumerate(zip(X, y)):
            h = self.nonlocal_size
            ind = D[i]<self.delta
            x = xi[ind]
            a, b = np.sum(ind[:i]),np.sum(ind[i+1:])
            if a<h:
                x = np.column_stack([np.full(a-h, m), x])
            elif a>h:
                x = x[a-h:]
            if b<h:
                x = np.column_stack([x, np.full(b-h, m)])
            elif b>h:
                x = x[b-h:]
            z[i] = x

        if X_ is None:
            z_ = z
        else:
            D_ = euclidean_distances(X_)
            z_ = np.empty((N, self.nonlocal_size))
            for i, (xi, yi) in enumerate(zip(X, y)):
                h = self.nonlocal_size
                ind = D[i]<self.delta
                x = xi[ind]
                a, b = np.sum(ind[:i]),np.sum(ind[i+1:])
                if a<h:
                    x = np.column_stack([np.full(a-h, m), x])
                elif a>h:
                    x = x[a-h:]
                if b<h:
                    x = np.column_stack([x, np.full(b-h, m)])
                elif b>h:
                    x = x[b-h:]
                z_[i] = x

        return normalize(self.kernel(X, X_) * self.y_kernel(z, z_), lb=_lb)

image_regressor = SelfEpanechnikovRegression(2, 300)

def compress(model, X, step=2, n_iter=1, block_size=(1,1)):
    from utils import image2data, data2image
    for _ in range(n_iter):
        X, shape, pos = image2data(X, block_size=block_size, with_pos=True)
        K = model._get_kernel(pos)
        X = model.kernel_predict(K, X)
        X = data2image(X, block_size=block_size, shape=shape, to_image=False)
        X = X[::step, ::step]
    return X


class MultiKernelLocalRegression(SelfLocalMeanRegression):

    def __init__(self, kernels=None, y_kernels=None, init_weights=None, **kwargs):
        self.init_weights = init_weights
        self.kernels = kernels
        self.y_kernels = y_kernels
        self.n_kernels = len(kernels)
        super().__init__(**kwargs)

    def fit(self, X, y):
        super().fit(X, y)
        if not hasattr(self, 'weights_'):
            self.compute_weights(X, y)

    def get_kernels(self, X, X_=None, y=None, y_=None):
        # override it, the core method of the class
        if X_ is None:
            X_ = getattr(self, 'X_train_', X)
        if y_ is None:
            y_ = getattr(self, 'y_train_', y)
        Ks = [kernel(X, X_) for kernel in self.kernels]
        if self.y_kernels is None or y is None:
            return Ks
        else:
            return Ks, [kernel(to_column_vector(y), to_column_vector(y_)) for kernel in self.y_kernels]

    def _get_kernel(self, X, X_=None, y=None, y_=None):
        if X_ is None:
            X_ = getattr(self, 'X_train_', X)
        if y_ is None:
            y_ = getattr(self, 'y_train_', y)
        if y is None:
            K = np.sum([w * K for K, w in zip(self.get_kernels(X, X_), self.weights_)], axis=0)
        else:
            K = np.sum([w * K * Ky for K, Ky, w in zip(*self.get_kernels(X, X_, y, y_), self.weights_)], axis=0)
        return normalize(K, lb=0.000001)

    def compute_weights(self, X, X_=None, y=None, y_=None, init_weights=None, *args, **kwargs):
        if X_ is None: X_ = X
        if y_ is None: y_ = y
        if init_weights is None:
            if self.init_weights is None:
                weight = np.ones(self.n_kernels)
            else:
                weight = self.init_weights
        else:
            weight = init_weights
        weight /= np.sum(weight)

        N = len(X)
        Xs = self.compute_stats(X, y)
        A = np.array([[np.sum(X * X1) for X1 in Xs] for X in Xs])
        for _ in range(100):
            weights -= 0.1*np.dot(A, weights)
            weights = np.maximum(weights, 0); weights /= np.sum(weights)
        self.weights_ = weights

    def compute_stats(self, X, y):
        return [normal_hollow(K * Ky) @ y for K, Ky in zip(*self.get_kernels(X, X, y, y))]

