"""
Fingerprint functions
"""


def binaryfp(row, feat_quants):
    """
    Function for creating a binary fingerprint based on feature quantiles
    when those feature quantiles are met.

    Ex:
    If player A has a 0.298 career batting average and 250 home runs
    and your representation is based upon:
    AVG > 0.250, AVG > 0.275, AVG > 0.300, HR > 200, HR > 400, HR > 600
    The fingerprint for player A will be:
    [1, 1, 0, 1, 0 ,0]

    Parameters
    ---------
    row: pandas row
        row of player information (e.g. AVG, OPS, etc.)
    feat_quants: dict
        dictionary of features and their quantiles

    Returns
    -------
    binary_fp: list
        binary fingerprint list
    """

    binary_fp = []
    for fkey, quants in feat_quants.items():
        val_bin = [0] * len(quants)
        for i in range(len(quants)):
            if row[fkey] >= quants[i]:
                # turn bit on only if larger than or equal to quant
                val_bin[i] = 1
        binary_fp.extend(val_bin)

    return binary_fp


def binnedfp(row, feat_quants):
    """
    Function for creating quantile bin fingerprints based on feature quantiles
    when only the highest matching quantile is stored. This creates a sparser
    fingerprint that avoids players matching on all the lower thresholds.

    Ex:
    If player A has a 0.298 career batting average and 250 home runs
    and your representation is based upon:
    AVG > 0.250, AVG > 0.275, AVG > 0.300, HR > 200, HR > 400, HR > 600
    The fingerprint for player A will be:
    [0, 1, 0, 1, 0 ,0]

    Parameters
    ---------
    row: pandas row
        row of player information (e.g. AVG, OPS, etc.)
    feat_quants: dict
        dictionary of features and their quantiles

    Returns
    -------
    binned_fp: list
        binned fingerprint list
    """
    binned_fp = []
    for fkey, quants in feat_quants.items():
        val_bin = [0] * len(quants)
        for i in range(len(quants) - 1, -1, -1):  # iterate backwards
            if row[fkey] >= quants[i]:
                val_bin[i] = 1
                break
        binned_fp.extend(val_bin)

    return binned_fp
