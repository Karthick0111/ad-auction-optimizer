"""
Replays a slice of the real (historical) holdout impression stream onto the
Kinesis bid-requests stream for one simulation run - this is what makes the
static Criteo holdout data feel "live" to the rest of the pipeline.

Invoked asynchronously by run_trigger. Reads the holdout JSONL file from S3
once per container (plain Python, deliberately no pandas/pyarrow - keeps
this function's Lambda layer far under the 250MB unzipped limit, which
pandas+pyarrow+numpy alone blow past), samples n_impressions rows (seeded,
reproducible via random.Random), and PutRecords them in batches of 500
(Kinesis's per-call limit) using run_id as the partition key so they all
land on one shard in order. Writes a sentinel record at the end so
bid_consumer knows the stream is finished.
"""
import json
import logging
import random

from common import HOLDOUT_S3_KEY, KINESIS_STREAM_NAME, S3_BUCKET, kinesis_client, s3_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_holdout = None
BATCH_SIZE = 500


def _load_holdout() -> list:
    global _holdout
    if _holdout is None:
        local_path = "/tmp/holdout.jsonl"
        s3_client().download_file(S3_BUCKET, HOLDOUT_S3_KEY, local_path)
        with open(local_path) as f:
            _holdout = [json.loads(line) for line in f]
        logger.info("Loaded holdout impression stream (%d rows) from s3://%s/%s", len(_holdout), S3_BUCKET, HOLDOUT_S3_KEY)
    return _holdout


def _build_record(run_id: str, sequence: int, row: dict) -> dict:
    features = {k: v for k, v in row.items() if k != "label"}
    return {
        "run_id": run_id,
        "sequence": sequence,
        "impression_id": f"{run_id}-{sequence}",
        "features": features,
        "true_label": int(row["label"]),
    }


def handler(event, context):
    run_id = event["run_id"]
    n_impressions = int(event["n_impressions"])
    seed = int(event.get("seed", 42))

    holdout = _load_holdout()
    rng = random.Random(seed)
    sample = rng.sample(holdout, k=min(n_impressions, len(holdout)))

    client = kinesis_client()
    for start in range(0, len(sample), BATCH_SIZE):
        chunk = sample[start:start + BATCH_SIZE]
        entries = [
            {"Data": json.dumps(_build_record(run_id, start + i, row)), "PartitionKey": run_id}
            for i, row in enumerate(chunk)
        ]
        client.put_records(StreamName=KINESIS_STREAM_NAME, Records=entries)
        logger.info("Put records %d-%d for run %s", start, start + len(chunk), run_id)

    client.put_record(
        StreamName=KINESIS_STREAM_NAME,
        PartitionKey=run_id,
        Data=json.dumps({"run_id": run_id, "is_sentinel": True}),
    )
    logger.info("Finished streaming %d impressions for run %s", len(sample), run_id)
    return {"statusCode": 200}
