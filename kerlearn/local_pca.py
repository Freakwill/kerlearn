#!/usr/bin/env python

import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import clone, TransformerMixin
from sklearn.metrics.pairwise import *
from scipy.spatial.distance import cdist


class WeightedPCA(PCA):
    """
    Weighted PCA
    """

    def __init__(self, weights=None, **kwargs):
        self.weights = weights
        super().__init__(**kwargs)

    def fit(self, X):
        if self.weights is None:
            return super().fit(X)
        else:
            return self.wfit(X, self.weights)

    def wfit(self, X, sample_weights):
        self.mean_ = np.dot(sample_weights, X) / np.sum(sample_weights)
        X_centerized = X - self.mean_
        return PCA.fit(self, sample_weights[:,None] * X_centerized)

    # def transform(self, X):
    #     return super().transform(X) / sample_weights[:,None]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))

    def shift(self, X, max_iter=16):
        for _ in range(max_iter):
            X = self.reconstruct(X)


class LocalPCA(PCA):
    """
    Local PCA
    """

    def __init__(self, kernel=rbf_kernel, strategy='reweighted', **kwargs):
        self.kernel = kernel
        self.strategy = strategy
        super().__init__(**kwargs)

    def fit_target(self, X, target):
        # fit at the target point
        weights = self.kernel([target], X)[0]
        return self.fit_target(X, widths)

    def wfit(self, X, weights):
        if self.strategy == 'reweighted':
            return WeightedPCA.wfit(self, X, sample_weights=weights)
        elif self.strategy == 'resampling':
            from scipy.stats import rv_discrete
            resamp = rv_discrete(name='resampling', values=(np.arange(len(X)), weights/np.sum(weights)))
            return PCA.fit(self, X[resamp.rvs(size=500)])
        else:
            raise ValueError("`strategy` should be `reweighted` or `resampling`")

    def reconstruct_target(self, X, target):
        self.fit_target(X, target)
        lc = self.components_
        lm = self.mean_
        return (target - lm) @ lc.T @ lc + lm

    def fit_targets(self, X, targets=None):
        if targets is None:
            targets = X
        weights = self.kernel(targets, X)
        self.mean_list_ = []
        self.components_list_ = []
        for w in weights:
            self.wfit(X, w)
            self.mean_list_.append(self.mean_)
            self.components_list_.append(self.components_)
        return self

    def reconstruct_targets(self, X, targets=None):
        if targets is None:
            targets = X
        self.fit_targets(X, targets)
        return np.array([(target - lm) @ lc.T @ lc + lm
            for (target, lm, lc) in zip(targets, self.mean_list_, self.components_list_)])

    def reconstruct(self, X):
        return self.reconstruct_targets(X)

    def error_target(self, X, target):
        return LA.norm(target - self.reconstruct_target(X, target))

    def error(self, X):
        return LA.norm(X - self.reconstruct(X))

    def fit(self, X):
        self.global_mean_ = np.mean(X, axis=0)
        self.global_componets_ = super().fit(X).components_
        return self

    def fit_transform_target(self, X, target):
        self.fit_target(X, target)
        lc = self.components_
        lm = self.mean_
        return (target - lm) @ lc.T + (lm - self.global_mean_) @ self.global_componets_.T

    def transform(self, X):
        return np.vstack([self.fit_transform_target(X, x) for x in X])

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class NeighborPCA(LocalPCA):
    """
    Neighour PCA
    """

    def __init__(self, n_neighours=0.05, **kwargs):
        self.n_neighours = n_neighours
        super().__init__(**kwargs)

    def get_neighbors(self, X, target):
        if self.n_neighours<1:
            self.n_neighours = int(self.n_neighours * len(X))
        return cdist([target], X)[0].argsort()[:self.n_neighours]

    def get_neighbors_targets(self, X, targets):
        if self.n_neighours<1:
            self.n_neighours = int(self.n_neighours * len(X))
        return cdist(targets, X).argsort()[:, :self.n_neighours]

    def fit_target(self, X, target):
        ind = self.get_neighbors(X, target)
        return self.nfit(X, ind)

    def fit_targets(self, X, targets=None):
        if self.n_neighours<1:
            self.n_neighours = int(self.n_neighours * len(X))
        if targets is None:
            targets = X
        inds = self.get_neighbors_targets(X, targets)
        self.mean_list_ = []
        self.components_list_ = []
        for ind in inds:
            self.nfit(X, ind)
            self.mean_list_.append(self.mean_)
            self.components_list_.append(self.components_)
        return self

    def nfit(self, X, ind):
        return PCA.fit(self, X[ind])


class MutualNeighborPCA(NeighborPCA):
    """
    Mutual Neighour PCA
    """

    def get_neighbors(self, X, target):
        if self.n_neighours<1:
            self.n_neighours = int(self.n_neighours * len(X))

        ind = cdist([target], X).ravel().argsort()[:self.n_neighours]
        return ind[cdist([target], X[ind])[0] < np.sort(cdist(X, X[ind]), axis=0)[self.n_neighours]]

    def get_neighbors_targets(self, X, targets):
        if self.n_neighours<1:
            self.n_neighours = int(self.n_neighours * len(X))

        inds = cdist(targets, X).argsort()[:, :self.n_neighours]
        res_inds = []
        for ind, target in zip(inds, targets):
            res_ind = ind[cdist([target], X[ind])[0] <= np.sort(cdist(X, X[ind]), axis=0)[self.n_neighours]]
            if len(res_ind) < self.n_components:
                res_ind = ind[:max(self.n_components+1, self.n_neighours//2)]
            res_inds.append(res_ind)
        return res_inds


