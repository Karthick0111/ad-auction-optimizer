"""
Pulls a manageable slice of the Criteo_x1 CTR dataset from Hugging Face (no
login/auth required - it's the same data behind the Criteo Kaggle Display
Advertising Challenge, mirrored publicly) and writes a cleaned local parquet
file, optionally uploading it to S3 for the offline training pipeline.

Runs once, locally, ahead of everything else - the deployed Lambda/Streamlit
pieces never talk to Hugging Face directly, only to the S3 artifact this
script produces.

Usage:
    python -m data.prepare_data --n-rows 200000 --s3-bucket my-bucket
    python -m data.prepare_data                    # local-only, no S3 upload
"""
import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from simulation.schema import CATEGORICAL_COLS, LABEL_COL, NUMERIC_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DATA_DIR / "criteo_sample.parquet"


def prepare(n_rows: int = 200_000) -> pd.DataFrame:
    from datasets import load_dataset

    logger.info("Downloading %d rows of reczoo/Criteo_x1 from Hugging Face", n_rows)
    ds = load_dataset("reczoo/Criteo_x1", split=f"train[:{n_rows}]")
    df = ds.to_pandas()

    # LightGBM's native categorical support needs pandas 'category' dtype -
    # avoids one-hot-encoding 26 high-cardinality columns into a sparse mess.
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "Prepared %d rows, %d columns, click rate %.4f",
        len(df), df.shape[1], df[LABEL_COL].mean(),
    )
    return df


def upload_to_s3(local_path: Path, bucket: str, key: str) -> None:
    import boto3

    logger.info("Uploading %s to s3://%s/%s", local_path, bucket, key)
    boto3.client("s3").upload_file(str(local_path), bucket, key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rows", type=int, default=200_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--s3-bucket", default=os.getenv("DATA_S3_BUCKET"))
    parser.add_argument("--s3-key", default="processed/criteo_sample.parquet")
    args = parser.parse_args()

    df = prepare(n_rows=args.n_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info("Wrote %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)

    if args.s3_bucket:
        upload_to_s3(args.output, args.s3_bucket, args.s3_key)
    else:
        logger.info("No --s3-bucket / DATA_S3_BUCKET set - skipping S3 upload (local-only run)")


if __name__ == "__main__":
    main()
