"""
Feature prep and generation methods
"""

import numpy as np


def generate_quantiles(df, stat_features):
    """
    Generate features and quantiles to use for fingerprinting

    Parameters
    ---------
    df: DataFrame
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


def feature_scaling(df, features, method="zscore"):
    """
    Generate feature scaling parameters for normalization

    Parameters
    ---------
    df: DataFrame
        dataframe to calc quantiles from
    features: list
        list of features to scale
    method: str
        method of scaling to use (minmax or zscore)

    Returns
    -------
    feat_scaling: dict
        dictionary of features and their scaling parameters
    """

    feat_scaling = {}
    for feat in features:
        if method == "minmax":
            min_v = df[feat].min()
            max_v = df[feat].max()
            feat_scaling[feat] = (min_v, max_v)
        elif method == "zscore":
            mean_v = df[feat].mean()
            std_v = df[feat].std()
            feat_scaling[feat] = (mean_v, std_v)
        else:
            raise ValueError("Invalid scaling method. Use 'minmax' or 'zscore'.")

    return feat_scaling
