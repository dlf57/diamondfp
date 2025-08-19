"""
Basic example of using diamondfp to create fingerprints and compare two players
"""

try:
    import polars as pl

    USE_POLARS = True
except ImportError:
    import pandas as pd

    USE_POLARS = False
from diamondfp.fingerprints import binaryfp, binnedfp
from diamondfp.utils.features import generate_quantiles
from diamondfp.scoring import jaccard, manhattan, cosine_sim

if USE_POLARS:
    df = pl.read_csv("../data/career-batting.csv")
else:
    df = pd.read_csv("../data/career-batting.csv")

stat_features = {
    "H": [0.5, 0.75, 0.9, 0.95],
    "2B": [0.75, 0.95],
    "3B": [0.75, 0.95],
    "HR": [0.9, 0.99],
    "K%": [0.1, 0.25],
    "BB%": [0.75, 0.99],
    "AVG": [0.5, 0.75, 0.9, 0.95],
    "OBP": [0.5, 0.75, 0.9, 0.95],
    "SLG": [0.5, 0.75, 0.9, 0.95],
    "OPS": [0.5, 0.75, 0.9, 0.95],
}

feat_quants = generate_quantiles(df, stat_features)


def get_player_row(df, player_name):
    if USE_POLARS:
        return df.filter(pl.col("Name") == player_name).to_dicts()[0]
    else:
        return df[df["Name"] == player_name].squeeze()


babe_ruth = binaryfp(get_player_row(df, "Babe Ruth"), feat_quants)
shohei_ohtani = binaryfp(get_player_row(df, "Shohei Ohtani"), feat_quants)
sim_score = jaccard(babe_ruth, shohei_ohtani)
print(f"Jaccard score: {sim_score:0.2f}")  # 0.72
cos_sim = cosine_sim(babe_ruth, shohei_ohtani)
print(f"Cosine similarity: {cos_sim:0.2f}")  # 0.85
man_dist = manhattan(babe_ruth, shohei_ohtani)
print(f"Manhattan distance: {man_dist}")  # 5
