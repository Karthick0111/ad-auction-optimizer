"""
Kinesis-triggered: for each bid-request event, scores CTR, has the bandit
pick a bid multiplier, simulates the second-price auction, updates
bandit/budget state, and writes the outcome to DynamoDB.

The Kinesis stream's partition key is run_id, so all events for one
simulation run land on the same shard in order - a single Lambda invocation
processes them sequentially, which is what keeps the read-modify-write on
bandit_state safe without needing distributed locking.

The CTR model is loaded once per warm container (module-level cache) and
reused across invocations - the standard Lambda cold-start optimization.
"""
import base64
import json
import logging
import random
import time
from decimal import Decimal

import lightgbm as lgb

from common import (
    DYNAMODB_BANDIT_TABLE,
    DYNAMODB_EVENTS_TABLE,
    DYNAMODB_RUNS_TABLE,
    MODEL_S3_KEY,
    S3_BUCKET,
    dynamodb_resource,
    s3_client,
)
from simulation.auction import draw_competitor_bids, settle_second_price
from simulation.bandit import ThompsonSamplingBandit
from simulation.schema import CATEGORICAL_COLS, NUMERIC_COLS, encode_categoricals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CATEGORY_MAPPINGS_S3_KEY = MODEL_S3_KEY.rsplit("/", 1)[0] + "/category_mappings.json"

_model = None
_category_mappings = None


def _load_model():
    global _model
    if _model is None:
        local_path = "/tmp/ctr_model.txt"
        s3_client().download_file(S3_BUCKET, MODEL_S3_KEY, local_path)
        _model = lgb.Booster(model_file=local_path)
        logger.info("Loaded CTR model from s3://%s/%s", S3_BUCKET, MODEL_S3_KEY)
    return _model


def _load_category_mappings():
    global _category_mappings
    if _category_mappings is None:
        local_path = "/tmp/category_mappings.json"
        s3_client().download_file(S3_BUCKET, CATEGORY_MAPPINGS_S3_KEY, local_path)
        with open(local_path) as f:
            _category_mappings = json.load(f)
        logger.info("Loaded category mappings from s3://%s/%s", S3_BUCKET, CATEGORY_MAPPINGS_S3_KEY)
    return _category_mappings


def _dec(x) -> Decimal:
    return Decimal(str(x))


def _mark_completed(runs_table, run_id: str) -> None:
    runs_table.update_item(
        Key={"run_id": run_id},
        UpdateExpression="SET #s = :s",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": "completed"},
    )


def handler(event, context):
    model = _load_model()
    category_mappings = _load_category_mappings()
    ddb = dynamodb_resource()
    runs_table = ddb.Table(DYNAMODB_RUNS_TABLE)
    events_table = ddb.Table(DYNAMODB_EVENTS_TABLE)
    bandit_table = ddb.Table(DYNAMODB_BANDIT_TABLE)

    for record in event["Records"]:
        payload = json.loads(base64.b64decode(record["kinesis"]["data"]))
        run_id = payload["run_id"]

        if payload.get("is_sentinel"):
            # Producer has finished streaming - flip status unless budget
            # exhaustion already completed the run earlier.
            run = runs_table.get_item(Key={"run_id": run_id}).get("Item")
            if run and run.get("status") == "running":
                _mark_completed(runs_table, run_id)
            continue

        run = runs_table.get_item(Key={"run_id": run_id}).get("Item")
        if run is None or run.get("status") != "running":
            continue  # run finished or budget already exhausted - drop the rest

        bandit_item = bandit_table.get_item(Key={"run_id": run_id}).get("Item")
        budget_remaining = float(bandit_item["budget_remaining"])
        if budget_remaining <= 0:
            _mark_completed(runs_table, run_id)
            continue

        bandit = ThompsonSamplingBandit.from_state(bandit_item["arm_state"])
        config = run["config"]
        value_per_click = float(config["value_per_click"])
        n_competitors = int(config["n_competitors"])
        mean_log_bid = float(config["mean_log_bid"])
        sigma = float(config["sigma"])

        encoded = encode_categoricals(payload["features"], category_mappings)
        feature_vector = [encoded[col] for col in NUMERIC_COLS + CATEGORICAL_COLS]
        predicted_ctr = float(model.predict([feature_vector])[0])

        arm = bandit.select_arm()
        bid = min(predicted_ctr * value_per_click * arm.multiplier, budget_remaining)

        rng = random.Random(payload["impression_id"])
        competitor_bids = draw_competitor_bids(n_competitors, mean_log_bid, sigma, rng)
        result = settle_second_price(bid, competitor_bids)

        clicked = bool(result.won and payload["true_label"] == 1)
        reward = (value_per_click if clicked else 0.0) - result.price_paid
        bandit.update(arm.multiplier, reward)

        new_budget_remaining = budget_remaining - result.price_paid
        bandit_table.update_item(
            Key={"run_id": run_id},
            UpdateExpression="SET arm_state = :a, budget_remaining = :b",
            ExpressionAttributeValues={
                ":a": {
                    mult: {k: _dec(v) for k, v in stats.items()}
                    for mult, stats in bandit.state().items()
                },
                ":b": _dec(new_budget_remaining),
            },
        )

        runs_table.update_item(
            Key={"run_id": run_id},
            UpdateExpression=(
                "ADD rows_processed :one, cumulative_spend :spend, "
                "cumulative_reward :reward, wins :win_inc, clicks :click_inc"
            ),
            ExpressionAttributeValues={
                ":one": 1,
                ":spend": _dec(result.price_paid),
                ":reward": _dec(reward),
                ":win_inc": 1 if result.won else 0,
                ":click_inc": 1 if clicked else 0,
            },
        )

        events_table.put_item(Item={
            "run_id": run_id,
            "sequence": int(payload["sequence"]),
            "arm_multiplier": _dec(arm.multiplier),
            "bid": _dec(bid),
            "won": result.won,
            "price_paid": _dec(result.price_paid),
            "clicked": clicked,
            "reward": _dec(reward),
            "predicted_ctr": _dec(predicted_ctr),
            "timestamp": int(time.time() * 1000),
        })

        if new_budget_remaining <= 0:
            _mark_completed(runs_table, run_id)

    return {"statusCode": 200}
