import numpy as np
import pandas as pd


def indexed_concat(lst):
    adjusted = [datum.copy() for datum in lst]
    for i, datum in enumerate(adjusted):
        datum.index = pd.MultiIndex.from_product([[i], datum.index])
    return pd.concat(adjusted, axis=0)
