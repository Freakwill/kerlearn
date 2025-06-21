#!/usr/bin/env python

"""Local Classifier/Mode

y^(X*) ~ max K(X*,X)
"""

import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import *

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(sparse_output=False)

from utils import normalize

from kerlearn import LazyMapMixin, LocalClassifierMixin

# def mode(x, c=None):
#     if c is None:
#         c = np.unique(x)
#     if len(x)==0:
#         k = np.random.randint(len(c))
#     else:
#         k = np.argmax([np.sum(x==k) for k in c])
#     return c[k]


class LocalClassifier(LocalClassifierMixin, BaseEstimator):

    def __init__(self, kernel=rbf_kernel, **kwargs):
        self.kernel = kernel

    def predict_proba(self, X, y=None):
        if y is None:
            y = self.y_train_
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(sparse_output=False)
        y = enc.fit_transform(y[:,None])
        K = self._get_kernel(X)
        return K @ y

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def get_centers(self):
        return np.row_stack([np.mean(self.X_train_[self.y_train_ == k], axis=0) for k in self.classes_])

    def inner_predict_proba(self, X, y=None):
        # inner-sample predicting
        if y is None:
            y = self.y_train_
        if y.ndim == 1:
            y = onehot.fit_transform(y[:,None])
        K = self.get_kernel(X, X_=None)
        return K @ y

    def inner_predict(self, X):
        # inner-sample predicting
        return np.argmax(self.inner_predict_proba(X), axis=1)


class NeighborClassifier(LocalClassifier):

    def __init__(self, threshold=1, *args, **kwargs):
        self.threshold = threshold
        super().__init__(*args, **kwargs)

    def _get_kernel(self, X, X_):
        return euclidean_distances(X, X_) < self.threshold


class LocalCluster(LazyMapMixin, LocalClassifier):

    def __init__(self, kernel=rbf_kernel, n_clusters=2, max_iter=20):
        self.kernel = kernel
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.classes_ = np.arange(n_clusters)

    def init_predict(self, X):
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=10)
        kmeans.fit(X)
        return kmeans.predict(X)

    @property
    def labels_(self):
        return self.y_train_

    def _init(self, X):
        self.X_train_ = X
        self.n_samples_ = len(X)


class LocalSoftCluster(LocalCluster):

    def init_predict_proba(self, X):
        y = self.init_predict(X)
        return onehot.fit_transform(y[:,None])

    def fit(self, X):
        self._init(X)
        y = self.init_predict_proba(X)
        for _ in range(self.max_iter):
            self._supervised_fit(X, y)
            y = self.inner_predict_proba(X, y)
        self.y_train_ = np.argmax(y, axis=1)
        return self


class NeighborCluster(LocalCluster):

    def __init__(self, threshold=1, *args, **kwargs):
        self.threshold = threshold
        super().__init__(*args, **kwargs)

    def _get_kernel(self, X, X_=None):
        if X_ is None:
            X_ = self.X_train_
        return euclidean_distances(X, X_) < self.threshold


class KNeighborsCluster(LocalCluster):

    def __init__(self, n_neighbors=5, *args, **kwargs):
        self.n_neighbors = n_neighbors
        super().__init__(*args, **kwargs)

    def _init(self, X):
        self.n_samples_ = X.shape[0]
        if self.n_neighbors<1:
            self.n_neighbors = int(np.round(n_samples * self.n_neighbors))
        self.X_train_ = X

    def _get_kernel(self, X, X_=None):
        if X_ is None:
            X_ = self.X_train_
        n = X.shape[0]
        neighbors = np.argsort(euclidean_distances(X, X_), axis=1)[:,:self.n_neighbors]
        K = np.zeros((n, self.n_samples_))
        for k, neighbors_ in zip(K, neighbors):
            k[neighbors_] = 1

        return normalize(K)


if __name__ == '__main__':

    from kernels import *

    _kernel = kernel_wrapper(rbf_kernel)

    from sklearn.datasets import make_moons
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    # X, y = make_moons(n_samples=400, noise=0.05, random_state=42)
    # X, X_test, y, y_test = train_test_split(X, y, test_size=0.3)

    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=500, centers=4, cluster_std=2, random_state=1)

    n_clusters = 4
    models = {
    'local-mode':LocalClassifier(kernel=_kernel(5)),
    # 'knn': KNeighborsClassifier(),
    'kmeans': KMeans(n_clusters=n_clusters),
    'local-clu': LocalCluster(kernel=_kernel(0.5), n_clusters=n_clusters, max_iter=15),
    'local-soft-clu': LocalSoftCluster(kernel=_kernel(1), n_clusters=n_clusters, max_iter=5),
    'neighbor-clu': NeighborCluster(threshold=0.2, n_clusters=n_clusters),
    'knn-clu': KNeighborsCluster(n_neighbors=30, n_clusters=n_clusters, max_iter=15),
    'knn-clf': KNeighborsClassifier(n_neighbors=10)
    }

    from utils import visualize
    import matplotlib.pyplot as plt
    plt.style.use('a_style.mplstyle')

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)

    def local_clf_demo():
        # from datasets import iris_2d
        # X, y = iris_2d(with_labels=True)

        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=300, noise=0.15, random_state=0)

        models['local-mode'].fit(X, y)
        # for name, model in models.items():
        #     model.fit(X, y)
        #     print(model.score(X_test, y_test))
        # raise
        visualize(ax, models['local-mode'], X, y, N1=60, N2=50, boundary=True, boundary_kw={'s':1.4, 'alpha':0.5, 'color':'k'}, scatter=False, background=True, background_kw={'alpha':0.15, 'edgecolor':'none'})
        visualize(ax, models['knn-clf'], X, y, N1=500, N2=400, boundary=True, boundary_kw={'s':1.6, 'alpha':0.75, 'color':'k'}, scatter_kw={'alpha': 0.6}, background=False)
                
        ax.plot([0], [0], ':k', linewidth=1.4, label='局部众数')
        ax.plot([0], [0], '-k', linewidth=1.6, label='KNN分类器')
        ax.legend()
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        plt.savefig('../src/knn-local-mode.png')
        plt.show()

    local_clf_demo()
    raise

    def local_clf_clu_demo():

        model = models['local-clu']

        models['kmeans'].fit(X, y)
        model.fit(X)
        y_ = model.inner_predict(X)

        visualize(ax, model, X, y_, N1=800, N2=600, boundary=True, boundary_kw={'s':1.6}, scatter=False, lim_ext=0.01)
        visualize(ax, models['kmeans'], X, y, N1=70, N2=70, boundary=True,
            boundary_kw={'s':1.8, 'alpha':0.5, 'edgecolor':'none'}, scatter_kw={'alpha': 0.7}, background=True, background_kw={'alpha':0.05, 'marker': 's'}, lim_ext=0.01)
                
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        # plt.savefig('../src/local-clf-clu-boundary.png')
        plt.show()

    local_clf_clu_demo()

