from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("nbastats")


def train_baseline_classifier(
    training_pkl: str | Path,
    out_dir: str | Path = "outputs",
    test_frac: float = 0.2,
    random_seed: int = 0,
) -> Tuple[Path, Dict[str, float]]:
    """Train a simple baseline classifier: predict whether home team wins (score_diff > 0).

    This is intentionally minimal: it demonstrates a clean, reviewable ML workflow rather than
    pushing SOTA performance.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    training_pkl = Path(training_pkl)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(training_pkl)

    y = (df["score_diff"] > 0).astype(int)
    # Keep only numeric features
    X = df.drop(columns=["score_diff", "awayTeam", "homeTeam", "gameIdx", "year"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=random_seed, stratify=y
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
    }

    out_path = out_dir / "baseline_logreg.pkl"
    pd.to_pickle(model, out_path)
    logger.info("Wrote model: %s", out_path)
    logger.info("Metrics: %s", metrics)
    return out_path, metrics
