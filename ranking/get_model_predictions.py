import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score

from ranking_model import (RankingLightFM, 
                           RankingBayesian,
                           RankingSequential,
                           validate_ranking_model)

reactions = pd.read_csv("./data/reactions.csv")
looks = reactions.image_id.unique()
users = reactions.user_id.unique()

approaches = [
    ("SeqTransformer", (lambda u, l, f: RankingSequential(u, l, f, {"device": "cuda:0", "model_type": "tr"}))),
    ("SeqRNN", (lambda u, l, f: RankingSequential(u, l, f, {"device": "cuda:0"}))),
    ("Bayesian", (lambda u, l, f: RankingBayesian(u, l, f, {"device": "cuda:0"}))),
    ("LightFM", (lambda u, l, f: RankingLightFM(u, l, f)))
]
feature_paths = [
    "./data/embeddings_base.csv",
    "./data/embeddings_segment-based-sim-ext.csv",
    "./data/embeddings_look-based.csv",
    "./data/embeddings_segment-based-sim.csv",
]

results = {}
for name, model_creator in approaches:
    results[name] = {}
    for feature_path in feature_paths:
        features_df = pd.read_csv(feature_path).set_index("image_id")
        if "embedding" in features_df.columns:
            item_features = features_df.loc[looks].embedding.values
            item_features = [[float(f) for f in fs[1:-1].split(",")] for fs in item_features] 
            item_features = np.array(item_features)

        model = model_creator(users, looks, item_features)
        gt, pred = validate_ranking_model(model, reactions, fit_every=100)
        roc_auc = roc_auc_score(gt, pred)
        print(name, feature_path, roc_auc)
        results[name][feature_path] = roc_auc

with open("/mnt/data/vnayok/embedder/validation_results_ranking/fr__2121.json", "w") as f:
    json.dump(results, f, indent=4)
