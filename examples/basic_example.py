import pandas as pd
from diamondfp.fingerprints import binaryfp, binnedfp
from diamondfp.utils.features import generate_quantiles
from diamondfp.scoring import jaccard, manhattan, cosine_sim

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

babe_ruth = binaryfp(df[df["Name"] == "Babe Ruth"].squeeze(), feat_quants)
shohei_ohtani = binaryfp(df[df["Name"] == "Shohei Ohtani"].squeeze(), feat_quants)
sim_score = jaccard(babe_ruth, shohei_ohtani)
print(f"Jaccard score: {sim_score:0.2f}")  # 0.72
cos_sim = cosine_sim(babe_ruth, shohei_ohtani)
print(f"Cosine similarity: {cos_sim:0.2f}")  # 0.85
man_dist = manhattan(babe_ruth, shohei_ohtani)
print(f"Manhattan distance: {man_dist}")  # 5
