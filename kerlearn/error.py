#!/usr/bin/env python

from sklearn.utils.validation import check_is_fitted


def check_x(model):
    check_is_fitted(model, 'X_train_'):


def check_y(model):
    check_is_fitted(model, 'y_train_'):
