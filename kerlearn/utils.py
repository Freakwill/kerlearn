
def _normalize(X, axis=1):
    # normalize a matrix
    if axis == 1:
        return X / X.sum(axis=1)[:,None]
    elif axis == 0:
        return X / X.sum(axis=0)
    else:
        raise ValueError('`axis` must be 0 or 1!')

def normalize(X, axis=1, lb=0):
    # normalize a matrix
    return _normalize(np.maximum(X, lb), axis=axis)
