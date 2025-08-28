"""
Feature prep and generation methods
"""

import numpy as np


def generate_quantiles(data, stat_features):
    """
    Generate features and quantiles to use for fingerprinting

    Parameters
    ---------
    data: dictionary
        dictionary  to calc quantiles from
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
            qv = np.quantile(data[feat], q)
            feat_v.append(qv)
        feat_quants[feat] = feat_v

    return feat_quants


def feature_scaling(data, features, method="zscore"):
    """
    Generate feature scaling parameters for normalization

    Parameters
    ---------
    data: dictionary
        dictionary to calc quantiles from
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
        values = np.array(data[feat])
        # use np.nan* to ignore NaNs in calculations
        if method == "minmax":
            min_v = float(np.nanmin(values))
            max_v = float(np.nanmax(values))
            feat_scaling[feat] = (min_v, max_v)
        elif method == "zscore":
            mean_v = float(np.nanmean(values))
            std_v = float(np.nanstd(values, ddof=0))
            feat_scaling[feat] = (mean_v, std_v)
        else:
            raise ValueError("Invalid scaling method. Use 'minmax' or 'zscore'.")

    return feat_scaling
