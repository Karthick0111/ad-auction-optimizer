"""
Shared feature-schema constants for the Criteo CTR dataset - used by data
prep, model training, and the producer Lambda alike, so column names live in
exactly one place. Deliberately dependency-free (no pandas/lightgbm import)
so the producer Lambda doesn't have to pull in the training stack just to
know which columns exist.
"""

NUMERIC_COLS = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
LABEL_COL = "label"


def encode_categoricals(features: dict, mappings: dict) -> dict:
    """Converts raw categorical string values to the integer codes LightGBM
    was trained on, using the category_mappings.json produced by
    train_ctr_model.py. Unseen values (not present in the training data)
    map to -1, which LightGBM treats like a missing value for categorical
    splits - a reasonable fallback rather than erroring on a category the
    model has never seen.

    Pure Python, no pandas - this is imported by the bid_consumer Lambda
    (which deliberately doesn't carry a pandas/pyarrow dependency) as well
    as the training script and the dashboard.
    """
    encoded = dict(features)
    for col in CATEGORICAL_COLS:
        col_mapping = mappings.get(col, {})
        encoded[col] = col_mapping.get(str(encoded.get(col)), -1)
    return encoded
