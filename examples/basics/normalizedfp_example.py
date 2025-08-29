"""
Example of using diamondfp to create normalized fingerprints and compare two players
"""

try:
    import polars as pl

    USE_POLARS = True
except ImportError:
    import pandas as pd

    USE_POLARS = False
from diamondfp.fingerprints import normalizedfp
from diamondfp.utils.features import feature_scaling
from diamondfp.scoring import cosine_sim

if USE_POLARS:
    df = pl.read_csv("../data/career-batting.csv")
else:
    df = pd.read_csv("../data/career-batting.csv")

features = [
    "HR",
    "K%",
    "BB%",
    "AVG",
    "OBP",
    "SLG",
    "OPS",
]

minmax_scaling = feature_scaling(df, features, method="minmax")
z_scaling = feature_scaling(df, features, method="zscore")


def get_player_row(df, player_name):
    if USE_POLARS:
        return df.filter(pl.col("Name") == player_name).to_dicts()[0]
    else:
        return df[df["Name"] == player_name].squeeze()


babe_ruth = get_player_row(df, "Babe Ruth")
shohei_ohtani = get_player_row(df, "Shohei Ohtani")

babe_minmax = normalizedfp(babe_ruth, minmax_scaling, method="minmax")
ohtani_minmax = normalizedfp(shohei_ohtani, minmax_scaling, method="minmax")
cos_sim = cosine_sim(babe_minmax, ohtani_minmax)
print(f"Cosine similarity w/ Min-Max Normalization: {cos_sim:0.2f}")  # 0.86

babe_zscore = normalizedfp(babe_ruth, z_scaling, method="zscore")
ohtani_zscore = normalizedfp(shohei_ohtani, z_scaling, method="zscore")
cos_sim = cosine_sim(babe_zscore, ohtani_zscore)
print(f"Cosine similarity w/ Z-score Normalization: {cos_sim:0.2f}")  # 0.95
