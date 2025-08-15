"""
Feature prep and generation methods
"""

import numpy as np


def generate_quantiles(df, stat_features):
    """
    Generate features and quantiles to use for fingerprinting

    Parameters
    ---------
    df: pandas DataFrame
        dataframe to calc quantiles from
    stat_features: dict
        dictionary containing stats and quantiles of interest

    Returns
    -------
    feat_quants: dict
        dictionary of features and their quantiles
    """

    feat_quants = {}
    features = stat_features.keys()
    for feat in features:
        quants = stat_features[feat]
        feat_v = []
        for q in quants:
            qv = np.quantile(df[feat], q)
            feat_v.append(qv)
        feat_quants[feat] = feat_v

    return feat_quants
