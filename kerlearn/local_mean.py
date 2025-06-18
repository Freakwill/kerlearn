#!/usr/local/bin/ python3

import numpy as np
from sklearn.metrics.pairwise import *
from scipy.linalg import toeplitz

from base import LocalModelMixin
from sklearn.metrics import r2_score as score
from kernels import *


def normal_hollow(K):
    np.fill_diagonal(K, 0)
    d = np.maximum(np.sum(K, axis=1), 0.0001)
    K /= d[:,None]
    np.fill_diagonal(K, -1)
    return K


def sq_sum(a):
    return np.sum(a**2)


def _toeplitz(a, T):
    L = len(a)
    if T>L:
        s = T//2+1
        a = np.insert(a, s, np.zeros(T-L))
    return toeplitz(a)


def time_kernel(t, s=None):
    if s is None:
        s = t
    s, t = np.meshgrid(s, t)
    return np.exp(np.minimum(10-2*np.abs(t-s), 0))


def weighted_mean(X, weight, lb=0):
    weight += lb
    return np.dot(weight, X) / np.sum(weight, axis=1)[:,None]


class LocalMeanMixin(LocalModelMixin):

    def fit(self, X, X_=None, fit_kernel=True):
        """local mean of time sequence:
            sum_s Ktsxs, K = K(X, X_)
        
        Args:
            X (array): target data
            X_ (array): training data
            h (int, optional): band-width
        """

        # self.kernel_matrix_ = self.get_kernel(X, X_=X_)  # K(X, X_)
        if X_ is None:
            X_ = X.copy()
        self.X_train_ = X_
        n_samples = X.shape[0]
        if fit_kernel:
            self.kernel_matrix_ = self._get_kernel(X, X_=X_)
        return self

    def fit_kernel(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        self.kernel_matrix_ = self._get_kernel(*args, **kwargs)
        return self

    def transform(self, X, *args, **kwargs):
        K = self.get_kernel(X, *args, **kwargs)
        if self.alpha ==1:
            return weighted_mean(self.X_train_, K)
        else:
            return self.alpha * weighted_mean(self.X_train_, K) + (1 - self.alpha) * X

    def reconstruct(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def shift(self, X, max_iter=5, *args, **kwargs):
        for _ in range(max_iter):
            X = self.transform(X, *args, **kwargs)
        return X

    def fit_shift(self, X, max_iter=5, *args, **kwargs):
        self.fit(X, *args, **kwargs)
        return self.shift(X, max_iter=5, *args, **kwargs)

    def score(self, X, X_=None):
        return score(X, self.reconstruct(X, X_=X_))


class TLocalMeanMixin(LocalModelMixin):

    def fit(self, X, X_=None, Ts=None, Ts_=None, fit_kernel=True):
        """local mean of time sequence:
            sum_s Ktsxs, K = K(X, X_)
        
        Args:
            X (array): target data
            X_ (array): training data
            h (int, optional): band-width
        """

        if X_ is None:
            X_ = X.copy()
        self.X_train_ = X_
        n_samples = X.shape[0]   
        if Ts is None:
            Ts = np.arange(n_samples)[:, None]
        if Ts_ is None:
            Ts_ = np.arange(X_.shape[0])[:, None]
        self.Ts_ = Ts_
        if fit_kernel:
            self.kernel_matrix_ = self._get_kernel(X, X_=X_, Ts=Ts, Ts_=Ts_)
        return self

    def transform(self, X, *args, **kwargs):
        K = self.get_kernel(X, *args, **kwargs)
        if self.alpha ==1:
            return weighted_mean(self.X_train_, K)
        else:
            return self.alpha * weighted_mean(self.X_train_, K) + (1 - self.alpha) * X

    def reconstruct(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def shift(self, X, max_iter=5, *args, **kwargs):
        for _ in range(max_iter):
            X = self.transform(X, *args, **kwargs)
        return X

    def fit_shift(self, X, max_iter=5, *args, **kwargs):
        self.fit(X, *args, **kwargs)
        return self.shift(X, max_iter=5, *args, **kwargs)

    def score(self, X, X_=None):
        return score(X, self.reconstruct(X, X_=X_))


class LocalMean(LocalMeanMixin):

    def __init__(self, kernel=rbf_kernel, time_kernel=None, alpha=1):
        self.kernel = kernel
        self.time_kernel = time_kernel
        self.alpha = alpha

    def _get_kernel(self, X, X_=None):
        if X_ is None: X_ = getattr(self, 'X_train_', X)
        return self.kernel(X, X_)


from kernels import kernel_wrapper, epanechnikov_kernel

def_kernel = kernel_wrapper(epanechnikov_kernel)
_rbf_kernel = kernel_wrapper(rbf_kernel)


class SeqLocalMean(TLocalMeanMixin):

    def __init__(self, kernel=rbf_kernel, time_kernel=None, alpha=1):
        self.kernel = kernel
        self.time_kernel = time_kernel
        self.alpha = alpha

    def get_kernel(self, X, X_=None, Ts=None, Ts_=None):
        # override it, the core method of the class
        if X_ is None: X_ = X
        if not hasattr(self, 'kernel_matrix_'):
            self.kernel_matrix_ = self._get_kernel(X, X_, Ts, Ts_)
        return self.kernel_matrix_

    def _get_kernel(self, X, X_=None, Ts=None, Ts_=None):
        # override it, the core method of the class
        if X_ is None:
            X_ = getattr(self, 'X_train_', X)
        if Ts is None:
            Ts = np.arange(X.shape[0])[:, None]
        if Ts_ is None:
            Ts_ = np.arange(X_.shape[0])[:, None]
        if self.kernel is None:
            KX = 1
        else:
            KX = self.kernel(X, X_)
        if self.time_kernel is None:
            return KX
        else:
            return KX * self.time_kernel(Ts, Ts_)

    def transform(self, X, X_=None, Ts=None, Ts_=None, *args, **kwargs):
        if X_ is None:
            X_ = getattr(self, 'X_train_', X)
        if Ts is None:
            Ts = np.arange(X.shape[0])[:, None]
        if Ts_ is None:
            Ts_ = np.arange(X_.shape[0])[:, None]
        K = self.get_kernel(X, X_, Ts, Ts_)
        if self.alpha ==1:
            return weighted_mean(self.X_train_, K)
        else:
            return self.alpha * weighted_mean(self.X_train_, K) + (1 - self.alpha) * X


class GaussianMean(SeqLocalMean):

    def __init__(self, width=1, time_width=1, *args, **kwargs):
        self.width = width
        self.time_width = time_width
        kernel = _rbf_kernel(1/width)
        if time_width is None:
            time_kernel = None
        else:
            time_kernel = _rbf_kernel(1/time_width)
        super().__init__(kernel=kernel, time_kernel=time_kernel, *args, **kwargs)


class EpanechnikovMean(SeqLocalMean):

    def __init__(self, width=1, time_width=1, *args, **kwargs):
        self.width = width
        self.time_width = time_width
        _kernel = make_kernel(epanechnikov_kernel, gamma=1/(width**2))
        if time_width is None:
            time_kernel = None
        else:
            time_kernel = make_kernel(epanechnikov_kernel, gamma=1/time_width**2)
        super().__init__(kernel=_kernel, time_kernel=time_kernel, *args, **kwargs)


class ManhattanMean(SeqLocalMean):

    def __init__(self, width=1, time_width=1, *args, **kwargs):
        self.width = width
        self.time_width = time_width
        _kernel = make_kernel(epanechnikov_kernel, gamma=1/(width**2))
        if time_width is None:
            time_kernel = None
        else:
            time_kernel = make_kernel(manhattan_kernel, gamma=1/time_width**2)
        super().__init__(kernel=_kernel, time_kernel=time_kernel, *args, **kwargs)


class KNNMean(SeqLocalMean):

    def __init__(self, n_neighbors=1, time_n_neighbors=4, weight=None, time_weight=None, *args, **kwargs):
        self.n_neighbors = n_neighbors
        self.time_n_neighbors = time_n_neighbors
        _kernel = make_kernel(weighted_knn_kernel, n_neighbors=n_neighbors, weight=weight)
        if time_n_neighbors is None:
            time_kernel = None
        else:
            time_kernel = make_kernel(weighted_knn_kernel, n_neighbors=time_n_neighbors, weight=time_weight)
        super().__init__(kernel=_kernel, time_kernel=time_kernel, *args, **kwargs)


class ConvolutionLocalMean(LocalMean):

    def __init__(self, *args, **kwargs):
        time_kernel = kwargs.get('time_kernel', None)
        if time_kernel is not None:
            assert isinstance(time_kernel, np.ndarray) and time_kernel.ndim == 1
        super().__init__(*args, **kwargs)

    def _TK(self, T):
        # time kernel matrixs
        return _toeplitz(self.time_kernel, T)


class AlgebraicLocalMean(LocalMean):

    def __init__(self, time_kernel=rbf_kernel, alpha=1):
        assert time_kernel is not None
        super().__init__(kernel=None, time_kernel=time_kernel, alpha=1)

    def get_kernel(self, X, X_=None, fix_kernel=False):
        # override it, the core method of the class
        if X_ is None: X_ = X
        if fix_kernel:
            return self.kernel_matrix_
        T, T_ = X.shape[0], X_.shape[0]
        return self.TK(T, T_)


class MultiKernelLocalMean(SeqLocalMean):

    def __init__(self, kernels=None, time_kernels=None, init_weights=None, *args, **kwargs):
        self.init_weights = init_weights
        self.kernels = kernels
        self.time_kernels = time_kernels
        self.n_kernels = len(kernels)
        super().__init__(*args, **kwargs)

    def fit(self, X, X_=None, *args, **kwargs):
        if not hasattr(self, 'weights_'):
            self.compute_weights(X, X_=None, *args, **kwargs)
        super().fit(X, X_=None, *args, **kwargs)

    def get_kernels(self, X, X_=None, fix_kernel=False):
        # override it, the core method of the class
        if X_ is None: X_ = X
        if fix_kernel:
            return self.kernel_matrix_
        Ks = [kernel(X, X_) for kernel in self.kernels]
        if self.time_kernels is None:
            return Ks
        else:
            T, T_ = X.shape[0], X_.shape[0]
            Ts = self._Ts(T)
            Ts_ = self._Ts(T_)
            return Ks, [kernel(Ts, Ts_) for kernel in self.time_kernels]

    def get_kernel(self, X, X_=None, fix_kernel=False):
        if X_ is None: X_ = X
        return np.mean([w * K * KT for K, KT, w in zip(*self.get_kernels, self.weights)], axis=0)

    def compute_weights(self, X, X_=None, init_weights=None, *args, **kwargs):
        if X_ is None: X_ = X
        if init_weights is None:
            if self.init_weights is None:
                w = np.random.random(self.n_kernels)
            else:
                w = self.init_weights
        else:
            w = init_weights
        Xs = self.compute_stats(X, X_) 
        A = np.array([[np.mean(X * X1) for X1 in Xs] for X in Xs]) / len(X_)**2
        w /= np.sum(w)
        for _ in range(10):
            w -= 0.0001 *np.dot(A, w)
            w = np.maximum(w, 0); w /= np.sum(w)
        self.weights_ = w

    def compute_stats(self, X, X_):
        Ks, KTs = self.get_kernels(X,X_)
        return [normal_hollow(K * KT) @ X_ for K, KT in zip(Ks, KTs)]

