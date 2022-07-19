import os
import argparse

import numpy as np

def recall(rs, N=10):
    """
    Ratio of recalled Ground Truth at topN Recalled Docs
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> recall(rs, N=1)
    0.333333
    >>> recall(rs, N=2)
    >>> 0.6666667
    >>> recall(rs, N=3)
    >>> 1.0
    Args:
        rs: Iterator of recalled flag()
    Returns:
        Recall@N
    """

    recall_flags = [np.sum(r[0:N]) for r in rs]
    return np.mean(recall_flags)