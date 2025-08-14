import numpy as np


def tanimoto(v1, v2):
    """
    Calculates the tanimoto coefficent between the two fingerprints
    Exact same as Jaccard score but tanimoto for those familiar with cheminformatics

    Parameters
    ---------
    v1: list
        fingerprint of player 1
    v2: list
        fingeprint of player 2

    Returns
    -------
    score: float
        tanimoto coefficient
    """
    v1 = np.array(v1, dtype=int)
    v2 = np.array(v2, dtype=int)
    c = np.sum(v1 & v2)
    u = np.sum(v1 | v2)
    return c / u if u != 0 else 0


def jaccard(v1, v2):
    """
    Calculates the Jaccard score between the two fingerprints
    Exact same as tanimoto but for those not familiar with cheminformatics

    Parameters
    ---------
    v1: list
        fingerprint of player 1
    v2: list
        fingeprint of player 2

    Returns
    -------
    score: float
        Jaccard score
    """
    v1 = np.array(v1, dtype=int)
    v2 = np.array(v2, dtype=int)
    c = np.sum(v1 & v2)
    u = np.sum(v1 | v2)
    return c / u if u != 0 else 0
