#!/usr/bin/env python

"""
Parameterized Kernels
"""


from sklearn.metrics.pairwise import *
from sklearn.metrics import DistanceMetric

def manhattan(*args, **kwargs):
    return pairwise_distances(metric='manhattan', *args, **kwargs)


def epanechnikov_kernel(x, y=None, gamma=1, coef=0):
    if y is None: y=x
    d = euclidean_distances(x, y, squared=True)
    return np.maximum(1 - d * gamma, 0) + coef


def manhattan_kernel(x, y=None, gamma=1, coef=0):
    if y is None: y=x
    d = manhattan(x, y)
    return np.maximum(1 - d * gamma, 0) + coef


def hollow_epanechnikov_kernel(coef=0, *args, **kwargs):
    K = epanechnikov_kernel(coef=coef, *args, **kwargs)
    return K * (K<(1+coef)) + coef


def neighbor_kernel(x, y=None, width=1):
    if y is None: y=x
    return euclidean_distances(x, y, squared=True) < width **2


def knn_kernel(x, y=None, n_neighbors=10):
    if y is None: y=x
    K = np.zeros((len(x), len(y)))
    if n_neighbors == 1:
        I = np.argmin(euclidean_distances(x, y, squared=True), axis=1)
        for i, a in enumerate(I):
            K[i, a] = 1
    else:
        I = np.argsort(euclidean_distances(x, y, squared=True), axis=1)
        for i, a in enumerate(I[:, :n_neighbors]):
            K[i, a] = 1
    return K


def weighted_knn_kernel(x, y=None, n_neighbors=10, weight=None):
    if y is None: y=x
    K = np.zeros((len(x), len(y)))
    if n_neighbors == 1:
        I = np.argmin(euclidean_distances(x, y, squared=True), axis=1)
        for i, a in enumerate(I):
            K[i, a] = 1
    else:
        I = np.argsort(euclidean_distances(x, y, squared=True), axis=1)
        for i, a in enumerate(I[:, :n_neighbors]):
            K[i, a] = 1
    if weight is None:
        return K
    else:
        return K * weight(x, y)


def kernel_wrapper(kernel=rbf_kernel):
    def make_kernel(epsilon=0, *args, **kwargs):
        def _f(X, Y=None):
            if Y is None: Y = X
            return kernel(X, Y, *args, **kwargs)+epsilon
        return _f
    return make_kernel


def make_kernel(kernel=rbf_kernel, epsilon=0, *args, **kwargs):
    def _f(X, Y=None):
        if Y is None:
            Y = X
        return kernel(X, Y, *args, **kwargs)+epsilon
    return _f


def kernelize(f):
    def kernel(x, y=None):
        if y is None:
            y = x
        return np.array([[f(a, b) for b in y] for a in x])
    return kernel


from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)

def make_kernel_with_label(kernel=rbf_kernel, Xlabels=None, Ylabels=None, *args, **kwargs):
    if Ylabels is None: Ylabels = Xlabels
    LX = onehot_encoder.fit_transform(labels[:,None])
    LY = onehot_encoder.fit_transform(labels[:,None])
    def _f(X, Y=None):
        X = np.column_stack((X, LX))
        Y = np.column_stack((X, LY))
        return kernel(X, Y, gamma=gamma)+epsilon
    return _f


class wrapper:

    rbf = kernel_wrapper(rbf_kernel)
    epanechnikov = kernel_wrapper(epanechnikov_kernel)
    polynomial = kernel_wrapper(polynomial_kernel)
    sigmoid = kernel_wrapper(sigmoid_kernel)
    manhattan = kernel_wrapper(manhattan_kernel)
    neighbor = kernel_wrapper(neighbor_kernel)
    knn = kernel_wrapper(knn_kernel)
    weighted_knn = kernel_wrapper(weighted_knn_kernel)

