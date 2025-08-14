import numpy as np


def gen_feat_quants(df, basic_feats, advanced_feats, basic_quants, advanced_quants):
    """
    Generate features and quantiles to use for fingerprinting

    Parameters
    ---------
    df: pandas DataFrame
        dataframe to calc quantiles from
    basic_feats: list
        basic feature list to use for the fingerprints
    advanced_feats: list
        advanced feature list to use for the fingerprints
    basic_quants: list
        list of quants for feature
    advanced_quants: list
        list of quants for feature

    Returns
    -------
    feat_quants: dict
        dictionary of features and their quantiles
    """
    feat_quants = {}

    features = basic_feats + advanced_feats
    for feat in features:
        if feat in basic_quants:
            quants = basic_quants
        else:
            quants = advanced_quants
        for q in quants:
            if feat == "K%" or feat == "BB%":
                if q == 0.9:
                    q = 0.1
                else:
                    q = 1.0 - q
            q_pct = int(q * 100)
            f_q = f"{feat}>{q_pct}"
            qv = np.quantile(df[feat], q)
            feat_quants[f_q] = float(qv)

    return feat_quants


def binary_fp(row, feat_quants):
    """
    Generate binary representation of whether the player's feature value is above the quantile

    Parameters
    ---------
    row: pandas row
        row of player information (e.g. AVG, OPS, etc.)
    feat_quants: dict
        dictionary of features and their quantiles

    Returns
    -------
    mlbin_fp: list
        binary fingerprint list
    """

    mlbin_fp = []
    for fkey in feat_quants.keys():
        if feat_quants[fkey] <= row[fkey.split(">")[0]]:
            # turn bit on if value is above quantile
            mlbin_fp.append(1)
        else:
            mlbin_fp.append(0)

    return mlbin_fp
