"""
Trains the CTR model on the prepared Criteo sample.

Two models, for comparison:
  - LightGBM (primary): native categorical support handles the 26
    high-cardinality columns directly, no one-hot blowup.
  - Logistic regression (baseline): categoricals go through the hashing
    trick (sklearn FeatureHasher) instead of one-hot, since C1-C26 have
    cardinalities in the thousands-to-millions - the same technique
    production CTR systems use at Criteo's actual scale.

SageMaker script-mode compatible: reads SM_CHANNEL_TRAIN / writes
SM_MODEL_DIR when running inside a SageMaker Training Job container, falls
back to local CLI args otherwise - the identical script runs in both places.

Also writes the holdout split back out (holdout.parquet) - those rows,
complete with real historical labels, become the "live" impression stream
the auction simulation replays through Kinesis.
"""
import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from simulation.schema import CATEGORICAL_COLS, LABEL_COL, NUMERIC_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HASH_FEATURES = 2**18  # ~262k buckets - bounded regardless of raw cardinality


def train_lightgbm(X_train, y_train, X_holdout, y_holdout):
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_COLS, free_raw_data=False)
    holdout_set = lgb.Dataset(X_holdout, label=y_holdout, categorical_feature=CATEGORICAL_COLS, reference=train_set)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "verbose": -1,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[holdout_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False), lgb.log_evaluation(period=0)],
    )
    preds = model.predict(X_holdout, num_iteration=model.best_iteration)
    return model, preds


def _hash_categoricals(df: pd.DataFrame) -> np.ndarray:
    hasher = FeatureHasher(n_features=HASH_FEATURES, input_type="string")
    rows = df[CATEGORICAL_COLS].astype(str).values.tolist()
    return hasher.transform(rows)


def train_logistic_baseline(X_train, y_train, X_holdout, y_holdout):
    from scipy.sparse import hstack

    numeric_train = X_train[NUMERIC_COLS].fillna(0).values
    numeric_holdout = X_holdout[NUMERIC_COLS].fillna(0).values
    hashed_train = _hash_categoricals(X_train)
    hashed_holdout = _hash_categoricals(X_holdout)

    features_train = hstack([numeric_train, hashed_train]).tocsr()
    features_holdout = hstack([numeric_holdout, hashed_holdout]).tocsr()

    model = LogisticRegression(max_iter=200, solver="liblinear")
    model.fit(features_train, y_train)
    preds = model.predict_proba(features_holdout)[:, 1]
    return model, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=os.environ.get("SM_CHANNEL_TRAIN", str(Path(__file__).resolve().parent.parent / "data" / "criteo_sample.parquet")))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", str(Path(__file__).resolve().parent)))
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_data_path = Path(args.train_data)
    if train_data_path.is_dir():
        # SageMaker channel directories can contain the file under any name
        candidates = list(train_data_path.glob("*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No .parquet file found in {train_data_path}")
        train_data_path = candidates[0]

    logger.info("Loading %s", train_data_path)
    df = pd.read_parquet(train_data_path)

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[LABEL_COL]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=args.holdout_frac, random_state=args.seed, stratify=y,
    )
    holdout_full = df.loc[X_holdout.index]

    # LightGBM's Python API can consume pandas 'category' dtype directly,
    # but that only works reliably when every consumer (training, the
    # dashboard, and the bid_consumer Lambda) shares the exact same
    # category->code mapping - fragile across process boundaries. Instead,
    # build the mapping explicitly from the train split and save it
    # (category_mappings.json) so every inference path applies the
    # identical encoding via simulation.schema.encode_categoricals().
    # Unseen categories (holdout or live-serving) fall back to -1.
    category_mappings = {
        col: {v: i for i, v in enumerate(sorted(X_train[col].astype(str).unique()))}
        for col in CATEGORICAL_COLS
    }

    def _encode(frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame.copy()
        for col in CATEGORICAL_COLS:
            encoded[col] = frame[col].astype(str).map(category_mappings[col]).fillna(-1).astype(int)
        return encoded

    X_train_encoded = _encode(X_train)
    X_holdout_encoded = _encode(X_holdout)

    logger.info("Training LightGBM on %d rows, evaluating on %d", len(X_train), len(X_holdout))
    lgb_model, lgb_preds = train_lightgbm(X_train_encoded, y_train, X_holdout_encoded, y_holdout)
    lgb_auc = roc_auc_score(y_holdout, lgb_preds)
    lgb_logloss = log_loss(y_holdout, lgb_preds)
    logger.info("LightGBM: AUC=%.4f LogLoss=%.4f", lgb_auc, lgb_logloss)

    logger.info("Training logistic-regression baseline (hashed categoricals)")
    lr_model, lr_preds = train_logistic_baseline(X_train, y_train, X_holdout, y_holdout)
    lr_auc = roc_auc_score(y_holdout, lr_preds)
    lr_logloss = log_loss(y_holdout, lr_preds)
    logger.info("LogisticRegression: AUC=%.4f LogLoss=%.4f", lr_auc, lr_logloss)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    lgb_model.save_model(str(model_dir / "ctr_model.txt"))
    joblib.dump(lr_model, model_dir / "baseline_model.pkl")
    with open(model_dir / "category_mappings.json", "w") as f:
        json.dump(category_mappings, f)

    metrics = {
        "lightgbm": {"auc": lgb_auc, "logloss": lgb_logloss, "best_iteration": lgb_model.best_iteration},
        "logistic_regression_baseline": {"auc": lr_auc, "logloss": lr_logloss},
        "n_train": len(X_train),
        "n_holdout": len(X_holdout),
        "click_rate": float(y.mean()),
    }
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote model artifacts and metrics.json to %s", model_dir)

    holdout_path = model_dir / "holdout.parquet"
    holdout_full.reset_index(drop=True).to_parquet(holdout_path, index=False)
    logger.info("Wrote holdout impression stream (%d rows) to %s", len(holdout_full), holdout_path)

    # Also written as JSONL: this is what the producer Lambda reads at
    # simulation time. Using plain JSON lines instead of parquet there
    # means producer needs zero pandas/pyarrow - just Python's stdlib json
    # module - which keeps its Lambda layer far under the 250MB unzipped
    # limit (pyarrow alone unzips to 200MB+).
    #
    # Categorical columns are cast to str column-wise (matching exactly how
    # category_mappings was built via X_train[col].astype(str)) BEFORE any
    # row-wise iteration - mixing float64 numeric columns and category-dtype
    # columns in a single row via .iterrows()/.itertuples() forces pandas to
    # upcast the whole row to a common dtype, which silently mangled
    # category values into "100.0"-style strings that then couldn't be
    # found in category_mappings (an easy, hard-to-notice bug: every
    # lookup would fall back to "unseen category" and it would still run,
    # just score everything worse).
    holdout_export = holdout_full.reset_index(drop=True).copy()
    for col in CATEGORICAL_COLS:
        holdout_export[col] = holdout_export[col].astype(str)

    jsonl_path = model_dir / "holdout.jsonl"
    records = holdout_export[NUMERIC_COLS + CATEGORICAL_COLS + [LABEL_COL]].to_dict(orient="records")
    with open(jsonl_path, "w") as f:
        for record in records:
            for col in NUMERIC_COLS:
                if pd.isna(record[col]):
                    record[col] = None
            record[LABEL_COL] = int(record[LABEL_COL])
            f.write(json.dumps(record) + "\n")
    logger.info("Wrote holdout impression stream (JSONL, %d rows) to %s", len(holdout_full), jsonl_path)


if __name__ == "__main__":
    main()
