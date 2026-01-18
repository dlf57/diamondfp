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


def create_archetypes(data, features, k=5, method="kmeans"):
    """
    Helper function to generate archetypes from a dataset.

    Parameters
    ---------
    data: pd.DataFrame
        DataFrame of player stats
    features: list
        List of feature names to use
    k: int
        Number of archetypes to find
    method: str
        Clustering method (currently only 'kmeans')

    Returns
    -------
    archetypes: pd.DataFrame
        DataFrame of archetype centroids
    """
    from sklearn.cluster import KMeans
    import pandas as pd
    
    X = data[features].dropna()
    
    if method == "kmeans":
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        
        # Create meaningful names or just indices
        archetypes = pd.DataFrame(centroids, columns=features)
        archetypes.index = [f"Archetype_{i}" for i in range(k)]
        
        return archetypes
    else:
        raise ValueError("Method not supported. Use 'kmeans'.")
