"""
Shared config/client helpers for the Lambda handlers. Packaged into the
Lambda Layer alongside simulation/ so bid_consumer, run_trigger, and
producer don't each duplicate this code.

All AWS resource names come from environment variables that Terraform wires
up per-function - nothing hardcoded, so the same code works in any AWS
account/region the infra gets deployed into.
"""
import os

import boto3

DYNAMODB_RUNS_TABLE = os.environ.get("DYNAMODB_RUNS_TABLE", "simulation_runs")
DYNAMODB_EVENTS_TABLE = os.environ.get("DYNAMODB_EVENTS_TABLE", "simulation_events")
DYNAMODB_BANDIT_TABLE = os.environ.get("DYNAMODB_BANDIT_TABLE", "bandit_state")
S3_BUCKET = os.environ.get("S3_BUCKET")
KINESIS_STREAM_NAME = os.environ.get("KINESIS_STREAM_NAME", "bid-requests")
MODEL_S3_KEY = os.environ.get("MODEL_S3_KEY", "models/latest/ctr_model.txt")
HOLDOUT_S3_KEY = os.environ.get("HOLDOUT_S3_KEY", "models/latest/holdout.jsonl")
PRODUCER_FUNCTION_NAME = os.environ.get("PRODUCER_FUNCTION_NAME", "ad-auction-producer")

MAX_IMPRESSIONS_PER_RUN = 5000  # bounds Lambda runtime + Kinesis/DynamoDB cost per demo run

_dynamodb = None
_s3 = None
_kinesis = None
_lambda_client = None


def dynamodb_resource():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb")
    return _dynamodb


def s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def kinesis_client():
    global _kinesis
    if _kinesis is None:
        _kinesis = boto3.client("kinesis")
    return _kinesis


def lambda_client():
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = boto3.client("lambda")
    return _lambda_client
