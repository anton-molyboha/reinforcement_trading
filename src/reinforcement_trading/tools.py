import numpy as np
import pandas as pd


def indexed_concat(lst):
    adjusted = [datum.copy() for datum in lst]
    for i, datum in enumerate(adjusted):
        datum.index = pd.MultiIndex.from_product([[i], datum.index])
    return pd.concat(adjusted, axis=0)


def random_round(a):
    shape = a.shape if hasattr(a, 'shape') else (1,)
    flr = np.floor(a).astype(np.int64, copy=False)
    return flr + (np.random.rand(*shape) < a - flr)


def logodds_to_probs(odds):
    scaled = np.asarray(odds) - np.expand_dims(np.max(odds, axis=-1), axis=-1)
    transformed = np.exp(scaled)
    return transformed / np.expand_dims(np.sum(transformed, axis=-1), axis=-1)


def weights_to_inds(weights, amplification=10):
    copy_counts = random_round(weights * amplification)
    res = np.empty(np.sum(copy_counts), dtype=np.int64)
    ptr = 0
    for i, cnt in enumerate(copy_counts):
        res[ptr:ptr+cnt] = i
        ptr += cnt
    return res
