
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.utils.extmath import safe_sparse_dot


class BaseEncoder(TransformerMixin):
    def encode(self, X):
        return self.transform(X)

    def decode(self, X):
        return self.inverse_transform(X)


def inplace_relu(X):
    np.maximum(X, 0, out=X)


class BaseMLPEncoder(BaseEncoder, BaseEstimator):

    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components
        super().__init__(hidden_layer_sizes=(n_components,), **kwargs)

    def transform(self, X):
        hidden_activation = inplace_relu
        activation = X
        activation = safe_sparse_dot(activation, self.coefs_[0]) + self.intercepts_[0]
        hidden_activation(activation)
        return activation

    def inverse_transform(self, Y):
        activation = Y
        activation = safe_sparse_dot(activation, self.coefs_[1]) + self.intercepts_[1]
        # output_activation(activation)
        return activation

    def fit(self, X, X_=None):
        if X_ is None:
            X_ = X
        return super().fit(X, X_)


class MLPEncoder(BaseMLPEncoder, MLPRegressor):

    pass


class MLPClassifierEncoder(BaseMLPEncoder, MLPClassifier):

    pass


if __name__ == '__main__':
    from datasets import fashion_images
    from utils import show_on_plain

    X, y = fashion_images(with_labels=True, ravel=True, channel=0)
    images = fashion_images()

    mlp = MLPClassifier(hidden_layer_sizes=(6,2), max_iter=1000)
    mlp.fit(X, y)
    print(mlp.score(X, y))
    raise
    Z = mlp.transform(X)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    show_on_plain(ax, images, Z=Z, y=y, n=20, alpha=0.4, s=20)

    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')
    plt.savefig('../src/mlp-clf-encoder-2d.png')
    plt.show()
