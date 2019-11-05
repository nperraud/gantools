import numpy as np

def KS_dist_1d(x,y):
    """Compute the KS distance between x and y."""
    assert(x.shape==y.shape)
    assert(len(x.shape)==1)
    np.testing.assert_almost_equal(np.sum(x),1)
    np.testing.assert_almost_equal(np.sum(y),1)
    return np.max(np.abs(np.cumsum(x)-np.cumsum(y)))