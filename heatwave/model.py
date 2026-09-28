from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from . import INPUT_COLUMNS, LABEL_MAP
from .pipeline import make_pipeline

K_VALUES = list(range(1, 26, 2))
MACRO_F1 = make_scorer(f1_score, labels=[0, 1, 2], average="macro", zero_division=0)


def split_by_year(features):
    years = pd.to_datetime(features.date).dt.year
    train = features.loc[years.between(2015, 2022)].copy()
    test = features.loc[years.between(2023, 2025)].copy()
    if set(pd.to_datetime(train.date).dt.year) != set(range(2015, 2023)) or test.empty:
        raise ValueError("Require all training years 2015-2022 and test observations in 2023-2025")
    return train, test


def tune_model(train, n_jobs=-1):
    years = pd.to_datetime(train.date).dt.year
    if not years.between(2015, 2022).all() or set(years) != set(range(2015, 2023)):
        raise ValueError("Tuning requires only and all training years 2015-2022")
    if min((years != y).sum() for y in years.unique()) < max(K_VALUES):
        raise ValueError("Too few observations per training fold for K=25")
    grid = {"knn__n_neighbors": K_VALUES, "knn__metric": ["euclidean", "manhattan"], "knn__weights": ["uniform", "distance"]}
    search = GridSearchCV(make_pipeline(), grid, scoring={"macro_f1": MACRO_F1, "accuracy": "accuracy"}, refit="macro_f1", cv=LeaveOneGroupOut(), n_jobs=n_jobs, return_train_score=True, error_score="raise")
    return search.fit(train[INPUT_COLUMNS], train.label, groups=years)


def save_artifact(pipeline, path, **metadata):
    artifact = dict(pipeline=pipeline, label_map=LABEL_MAP, input_columns=INPUT_COLUMNS, sklearn_version=sklearn.__version__, metadata=metadata)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return artifact


def load_artifact(path):
    """Load trusted local artifacts only; joblib deserialization executes Python."""
    artifact = joblib.load(path)
    required = {"pipeline", "label_map", "input_columns", "sklearn_version"}
    if not required <= artifact.keys():
        raise ValueError("Artifact does not satisfy the handoff contract")
    if artifact["sklearn_version"] != sklearn.__version__:
        raise ValueError(f"Artifact needs scikit-learn {artifact['sklearn_version']}; installed {sklearn.__version__}")
    return artifact


def neighbor_breakdown(pipeline, raw):
    """Counts and weighted votes, including sklearn's zero-distance convention."""
    if len(raw) != 1:
        raise ValueError("Pass exactly one observation")
    knn = pipeline.named_steps["knn"]
    distances, indices = knn.kneighbors(pipeline[:-1].transform(raw))
    # sklearn encodes fitted targets in _y; map back through classes_.
    labels = knn.classes_[knn._y[indices[0]].astype(int)]
    distances = distances[0]
    if knn.weights == "uniform":
        votes = np.ones(len(distances))
    elif np.any(distances == 0):
        votes = (distances == 0).astype(float)
    else:
        votes = 1 / distances
    neighbors = pd.DataFrame({"training_index": indices[0], "label": labels, "severity": [LABEL_MAP[int(x)] for x in labels], "distance": distances, "vote_weight": votes})
    breakdown = pd.DataFrame(index=list(LABEL_MAP.values()))
    breakdown["neighbors"] = neighbors.groupby("severity").size().reindex(breakdown.index, fill_value=0)
    breakdown["vote_share"] = neighbors.groupby("severity").vote_weight.sum().reindex(breakdown.index, fill_value=0) / votes.sum()
    return breakdown, neighbors
