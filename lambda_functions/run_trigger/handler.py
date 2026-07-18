"""
Starts a new simulation run: writes the run config + fresh bandit state to
DynamoDB, then asynchronously invokes the producer Lambda to start streaming
impressions onto Kinesis. Returns immediately with the new run_id so the
caller (the Streamlit app) can start polling DynamoDB right away rather than
waiting for the whole simulation to finish.
"""
import json
import logging
import uuid
from decimal import Decimal

from common import (
    DYNAMODB_BANDIT_TABLE,
    DYNAMODB_RUNS_TABLE,
    MAX_IMPRESSIONS_PER_RUN,
    PRODUCER_FUNCTION_NAME,
    dynamodb_resource,
    lambda_client,
)
from simulation.bandit import DEFAULT_ARMS, ThompsonSamplingBandit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _dec(x) -> Decimal:
    return Decimal(str(x))


def handler(event, context):
    body = event.get("body", event)  # supports both direct invoke and API Gateway proxy payloads
    if isinstance(body, str):
        body = json.loads(body)

    budget = float(body["budget"])
    value_per_click = float(body["value_per_click"])
    n_competitors = int(body.get("n_competitors", 5))
    mean_log_bid = float(body.get("mean_log_bid", 0.0))
    sigma = float(body.get("sigma", 0.4))
    n_impressions = min(int(body.get("n_impressions", 2000)), MAX_IMPRESSIONS_PER_RUN)
    seed = int(body.get("seed", 42))

    run_id = str(uuid.uuid4())
    ddb = dynamodb_resource()
    runs_table = ddb.Table(DYNAMODB_RUNS_TABLE)
    bandit_table = ddb.Table(DYNAMODB_BANDIT_TABLE)

    runs_table.put_item(Item={
        "run_id": run_id,
        "status": "running",
        "config": {
            "budget": _dec(budget),
            "value_per_click": _dec(value_per_click),
            "n_competitors": n_competitors,
            "mean_log_bid": _dec(mean_log_bid),
            "sigma": _dec(sigma),
            "n_impressions": n_impressions,
            "seed": seed,
        },
        "rows_processed": 0,
        "cumulative_spend": _dec(0),
        "cumulative_reward": _dec(0),
        "wins": 0,
        "clicks": 0,
    })

    empty_bandit = ThompsonSamplingBandit(arms=DEFAULT_ARMS)
    bandit_table.put_item(Item={
        "run_id": run_id,
        "budget_remaining": _dec(budget),
        "arm_state": {
            mult: {k: _dec(v) for k, v in stats.items()}
            for mult, stats in empty_bandit.state().items()
        },
    })

    lambda_client().invoke(
        FunctionName=PRODUCER_FUNCTION_NAME,
        InvocationType="Event",  # async - producer streams in the background
        Payload=json.dumps({"run_id": run_id, "n_impressions": n_impressions, "seed": seed}),
    )

    logger.info("Started run %s (%d impressions, budget=%.2f)", run_id, n_impressions, budget)
    return {"statusCode": 200, "body": json.dumps({"run_id": run_id})}
